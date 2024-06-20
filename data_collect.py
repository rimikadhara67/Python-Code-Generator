from curtsies.fmtfuncs import red, green, bold
from github import Github
import time
from datetime import datetime
import os

class DataCollector:
    def __init__(self, access_token_file="token.txt"):
        self.access_token = open(access_token_file, "r").read().strip()
        self.github_api = Github(self.access_token)

    def collect_and_clean_repos(self, days_back=6):
        end_time = time.time() - 86400 * 3
        start_time = end_time - 86400 * days_back

        for i in range(days_back):
            try:
                start_time_str = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%d')
                end_time_str = datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%d')
                query = f"language:python created:{start_time_str}..{end_time_str}"
                print(f"Querying for repos created from {start_time_str} to {end_time_str}")
                end_time -= 86400
                start_time -= 86400

                result = self.github_api.search_repositories(query)
                print(f"Found {result.totalCount} repositories")

                for repo in result:
                    repo_dir = f"repos/{repo.owner.login}/{repo.name}"
                    print(f"Cloning {repo.clone_url} into {repo_dir}")

                    os.makedirs(repo_dir, exist_ok=True)
                    os.system(f"git clone {repo.clone_url} {repo_dir}")

                    # Remove non-Python files
                    for dirpath, _, filenames in os.walk(repo_dir):
                        for f in filenames:
                            full_path = os.path.join(dirpath, f)
                            if full_path.endswith(".py"):
                                print(green(f"Keeping: {full_path}"))
                            else:
                                # print(red(f"Deleting: {full_path}"))
                                os.remove(full_path)

            except Exception as e:
                print(str(e))
                print(red(bold("Broke for some reason....")))
                time.sleep(120)

        print(f"Finished, your new time is: {start_time}")
