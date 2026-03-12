import warnings

# Suppress third-party deprecation warnings from pysbd and pydantic/crewai dependencies.
# These must be set here (earliest import point) AND we override showwarning because
# crewai's import chain resets warning filters.
_original_showwarning = warnings.showwarning

_SUPPRESSED_MODULES = ("pysbd", "pydantic")


def _filtered_showwarning(message, category, filename, lineno, file=None, line=None):
    if any(mod in filename for mod in _SUPPRESSED_MODULES):
        return
    _original_showwarning(message, category, filename, lineno, file, line)


warnings.showwarning = _filtered_showwarning
