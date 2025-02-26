import inspect
import os
import subprocess
import functools
from typing import Any, Callable, TypeVar, cast, Optional

# Create a generic type variable for the function
F = TypeVar("F", bound=Callable[..., Any])


def github_permalink(func: F) -> F:
    """
    Decorator that generates a GitHub permalink for the decorated function
    and makes it available as a 'github_permalink' attribute.

    Example usage:
        @github_permalink
        def my_function():
            # Get permalink from within the function
            permalink = get_current_permalink()
            print(f"This function's permalink: {permalink}")
    """
    # Get the source file and line number of the function
    source_file: Optional[str] = inspect.getsourcefile(func)
    if not source_file:
        setattr(func, "github_permalink", None)
        return func

    # Convert to absolute path
    source_file = os.path.abspath(source_file)

    # Get the function's line number (the line where the function is defined)
    _, func_lineno = inspect.getsourcelines(func)

    try:
        # Get the Git repository root
        repo_root: str = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=os.path.dirname(source_file),
            text=True,
        ).strip()

        # Get the current commit hash
        commit_hash: str = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(source_file),
            text=True,
        ).strip()

        # Get the relative file path from repo root
        rel_path: str = os.path.relpath(source_file, repo_root)

        # Get the remote URL
        remote_url: str = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=os.path.dirname(source_file),
            text=True,
        ).strip()

        # Convert Git URL to GitHub URL format
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]

        if remote_url.startswith("git@github.com:"):
            github_url: str = f"https://github.com/{remote_url[15:]}"
        elif remote_url.startswith("https://github.com/"):
            github_url = remote_url
        else:
            # For other formats, just use the URL as is
            github_url = remote_url

        # Create permalink
        permalink: str = (
            f"{github_url}/blob/{commit_hash}/{rel_path}#L{func_lineno}"
        )

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not in a git repository or git not available
        permalink = f"file://{source_file}#L{func_lineno}"

    # Attach permalink to function
    setattr(func, "github_permalink", permalink)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    setattr(wrapper, "github_permalink", permalink)
    return cast(F, wrapper)


def get_current_permalink() -> Optional[str]:
    """
    Get the GitHub permalink of the currently executing function.
    Must be called from within a function decorated with @github_permalink.

    Returns:
        The GitHub permalink as a string, or None if not available.
    """
    # Get the frame of the calling function
    current_frame = inspect.currentframe()
    if current_frame is None:
        return None

    # Get the frame of the function that called get_current_permalink()
    caller_frame = current_frame.f_back
    if caller_frame is None:
        return None

    # Get the function object from the frame
    func_name = caller_frame.f_code.co_name
    if func_name not in caller_frame.f_globals:
        # Try to find it in the locals (for nested functions)
        for value in caller_frame.f_locals.values():
            if (
                hasattr(value, "__name__")
                and value.__name__ == func_name
                and hasattr(value, "github_permalink")
            ):
                return value.github_permalink

        # If not found, look through globals for functions with the same code object
        for value in caller_frame.f_globals.values():
            if (
                hasattr(value, "__code__")
                and value.__code__ == caller_frame.f_code
                and hasattr(value, "github_permalink")
            ):
                return value.github_permalink

        return None

    # Get the function from globals
    func = caller_frame.f_globals[func_name]

    # Return the permalink if available
    if hasattr(func, "github_permalink"):
        return func.github_permalink

    return None
