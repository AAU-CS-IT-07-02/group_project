# group_project
Smart Building Control. Semester project for AAU's CS IT

## Documentation of the whole project is managed by MkDocs

1. Ensure that your python venv is ready by: 
    - run ```python3 -m venv venv```
    - run ```.\venv\Scripts\Activate.ps1``` for Windows PowerShell
    - run `> cd venv/Scripts` and `> activate` for command prompt  
    - run ```source venv/bin/activate``` for linux or macOS
    - install the requirements using the command ```pip install -r requirements.txt```

2. Run the mkdocs server with:
```bash
mkdocs serve -a 0.0.0.0:<free-port>
```

## Guidelines for Writing Python Code (for Documentation with MkDocs)

To ensure that our code examples and project modules are easy to understand and look consistent across the documentation, please follow these conventions when writing Python code.

### 1. General Style

* Follow **PEP 8** for code formatting (indentation, naming, line length, etc.).
* Use **clear, descriptive variable and function names**.
* Keep each script or function **focused on a single purpose**.
* Use **type hints** wherever possible:

  ```python
  def compute_area(radius: float) -> float:
      return 3.1415 * radius ** 2
  ```

### 2. Documentation & Comments

* Every public function, class, or module should include a **docstring** following the [PEP 257](https://peps.python.org/pep-0257/) conventions.
* Include a brief description, arguments, return values, and (if relevant) possible exceptions:

  ```python
  def greet(name: str) -> str:
      """
      Return a friendly greeting.

      Args:
          name: The name of the person to greet.

      Returns:
          A formatted greeting string.
      """
      return f"Hello, {name}!"
  ```

### 3. Code Examples in Documentation

When adding code examples to Markdown files (e.g. for MkDocs):

* Use **fenced code blocks** with a language identifier for syntax highlighting:

  ````markdown
  ```python
  def hello():
      print("Hello from MkDocs!")
  ````

  ```
  ```
* Keep examples short and self-contained.
* Show both the **function** and a **simple usage example** if appropriate.

### 4. Admonitions and Layout (optional)

If you’re adding examples inside notes, tips, or other MkDocs callouts, you can use **`pymdownx.superfences`** (already enabled in `mkdocs.yml`):

````markdown
!!! example "Usage Example"
    ```python
    result = greet("Alice")
    print(result)
    ```
````

### 5. Auto-Generated API Docs

Structure and document the files properly so they can be integrated with **`mkdocstrings`**:

```markdown
::: my_package.my_module
```

This will automatically pull in your docstrings into the rendered site.

