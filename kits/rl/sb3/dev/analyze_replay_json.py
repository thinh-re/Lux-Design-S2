import json


def decode_replay_file(replay_file: str):
    with open(replay_file, "r") as f:
        replay = json.load(f)
    print()


if __name__ == "__main__":
    decode_replay_file(replay_file="replay.json")
