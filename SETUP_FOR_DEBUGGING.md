# How to setup for debugging

Do not run `pip install --upgrade luxai_s2`

Instead, using these commmands to setup for debugging in vscode.

```bash
cd luxai_s2
pip install -e .
cd ..
```

Or run

```bash
python luxai_s2/luxai_runner/cli.py kits/rl/sb3/main.py kits/python/main.py -v 2 -o replay.json
```
