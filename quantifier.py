import importlib
import inspect
from typing import Any, Dict, List, Union


class Quantifier:
    """
    UncertaintyZoo Quantifier
    -------------------------
    A unified interface for computing uncertainty across
    diverse methods and model architectures (CodeBERT, ChatGLM, etc.).

    Example
    -------
        from uncertainty import Quantifier
        uq = Quantifier(model, methods=["mc_dropout_variance", "predictive_entropy"])
        score = uq.quantify(
            input_text="int main(){ char buf[8]; gets(buf); }",
            prompt="Determine if the following code is vulnerable.",
            model_type="generative",
            num_samples=10
        )
    """

    def __init__(self, model, methods: Union[str, List[str]]):
        """
        Initialize the Quantifier.

        Parameters
        ----------
        model : transformers.PreTrainedModel
            The underlying model instance (e.g., CodeBERT, ChatGLM, Qwen).
        methods : str or List[str]
            One or more uncertainty method names, corresponding to files
            under `methods/` (without the .py extension).
        """
        if isinstance(methods, str):
            methods = [methods]
        self.model = model
        self.methods = methods
        self.loaded_methods = {}
        self._load_methods()

    # ------------------------------------------------------------
    def _load_methods(self):
        """Dynamically import all specified methods."""
        for m in self.methods:
            try:
                module_path = f"methods.{m}"
                self.loaded_methods[m] = importlib.import_module(module_path)
            except ModuleNotFoundError:
                raise ImportError(f"[UncertaintyZoo] Method '{m}' not found in methods/ directory.")

    # ------------------------------------------------------------
    def quantify(
        self,
        input_text: str,
        prompt: str = "",
        model_type: str = "generative",
        task_type: str = "classification",
        num_samples: int = 10,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute uncertainty for a given input text or prompt.

        Parameters
        ----------
        input_text : str
            Input text or code snippet to evaluate.
        prompt : str, optional
            Optional task prompt (especially for generative models).
        model_type : {"generative", "discriminative"}
            Determines how the method is executed.
        task_type : str
            Task category, e.g., "classification", "generation", "reasoning".
        num_samples : int
            Number of stochastic forward passes (used by ensemble or MC methods).
        **kwargs :
            Additional keyword arguments passed to each uncertainty method.

        Returns
        -------
        dict
            {method_name: uncertainty_score}
        """
        results = {}

        for m_name, module in self.loaded_methods.items():
            # Identify the callable function within the module
            func = None
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                # convention: function has the same name as file, or include 'uncertainty'
                if name.lower() in [m_name.lower(), "uncertainty", m_name.split("_")[0]]:
                    func = obj
                    break
            if func is None:
                # fallback: first callable function
                funcs = [obj for _, obj in inspect.getmembers(module, inspect.isfunction)]
                if funcs:
                    func = funcs[0]
                else:
                    raise ValueError(f"[UncertaintyZoo] No callable found in {m_name}.py")

            # Build argument dict dynamically
            args = {}
            sig = inspect.signature(func)
            for param in sig.parameters:
                if param == "model":
                    args["model"] = self.model
                elif param == "text":
                    args["text"] = input_text
                elif param == "prompt":
                    args["prompt"] = prompt
                elif param == "model_type":
                    args["model_type"] = model_type
                elif param == "num_samples":
                    args["num_samples"] = num_samples
                elif param in kwargs:
                    args[param] = kwargs[param]

            # Call the method and store result
            try:
                result = func(**args)
                # standardize output
                if isinstance(result, dict):
                    score = result.get("score", result.get("uncertainty", 0.0))
                else:
                    score = float(result)
                results[m_name] = float(score)
            except Exception as e:
                print(f"[UncertaintyZoo] Error running {m_name}: {e}")
                results[m_name] = None

        return results
