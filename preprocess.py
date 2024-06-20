from curtsies.fmtfuncs import red
import os

class Preprocessor:
    def __init__(self):
        self.max_char_length = 512
        self.min_char_length = 400
        self.newline_char = "<N>"
        self.repo_directory = "repos"
        self.data_file = "data.txt"

    def process_repos(self):
        full_paths = []
        for dirpath, _, filenames in os.walk(self.repo_directory):
            for f in filenames:
                full_path = os.path.join(dirpath, f)
                full_paths.append(full_path)

        with open(self.data_file, "a") as f:
            for fpath in full_paths:
                try:
                    d = open(fpath, "r").read()
                    fd = d.replace("\n", self.newline_char)

                    if 100 < len(d) <= self.max_char_length:
                        f.write(fd + "\n")

                    else:
                        sd = fd.split(f"{self.newline_char}{self.newline_char}")
                        substring = ""
                        for split in sd:
                            substring += split + f"{self.newline_char}{self.newline_char}"
                            if self.min_char_length <= len(substring) <= self.max_char_length:
                                f.write(substring + "\n")
                                substring = ""

                except Exception as e:
                    print(red(str(e)))
