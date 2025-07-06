# AI Influencer Agency
主要可以参考```src/prompt.py```里的prompt。

### Launch the Project
1. Launch mcp server using ```uv run uvicorn src/mcp:app --reload```
2. Run agent using ```uv run src/onboarding.py``` and ```uv run src/planninng.py```

### Using UV To Setup the Project
[Documentation](https://docs.astral.sh/uv/)
1. install uv on the local environment ```uv python install 3.12```
2. clone and cd into this project
3. create virtual environment in the project folder ```uv venv --python 3.12```
4. run ```uv sync``` to install local python virtual environment and dependencies
5. ```source .venv/bin/activate``` or ```.venv\Scripts\activate``` to activate virtual environment
6. To install dependencies, run ```uv add```, ```uv pip install```. Use ```uv remove``` to remove project
7. To sync dependencies according to a particular file, run ```uv pip sync file.py```
8. To run python sciptes, use ```uv run main.py```
9. Run ```uv cache clean``` to clean up cache files
