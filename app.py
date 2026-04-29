from roadsentinel.ui import build_demo, get_launch_kwargs


demo = build_demo()


if __name__ == "__main__":
    demo.launch(**get_launch_kwargs())
