import re

with open("pyproject.toml", "r") as f:
    content = f.read()

def replacer(match):
    major, minor, patch = map(int, match.group(1).split('.'))
    new_version = f"{major}.{minor}.{patch + 1}"
    print(f"Bumping version to {new_version}")
    return f'version = "{new_version}"'

new_content = re.sub(r'version = "([^"]+)"', replacer, content)

with open("pyproject.toml", "w") as f:
    f.write(new_content)