# create config_copy.py

import os


def create_config_copy():
    if os.path.exists("config_files/config_copy.py"):
        os.remove("config_files/config_copy.py")

    with open("config_files/config.py", "r", encoding="utf-8") as f:
        configg = f.read()

    with open("config_files/user_config.py", "r", encoding="utf-8") as f:
        u_configg = f.read()

    with open("config_files/input_list.py", "r", encoding="utf-8") as f:
        input_listg = f.read()


    with open("config_files/config_copy.py", "w", encoding="utf-8") as f:
        f.write(configg)
        f.write("\n\n# user config\n")
        f.write(u_configg)
        f.write("\n\n# input list\n")
        f.write(input_listg)