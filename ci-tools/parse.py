import argparse
import os

def set_env_variable_from_string(input_string):
    if "ci-exactly:" in input_string:
        test_name = input_string.split("ci-exactly:")[1].strip()
        os.environ["TEST_FILE"] = test_name
        print(f'Successfully set TEST_FILE to: {test_name}')
    else:
        print('The PR body does not contain "ci-exactly:" so running all tests')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set TEST_FILE environment variable from PR body")
    parser.add_argument("--test", required=True, help="PR body containing 'ci-exactly:'")

    args = parser.parse_args()
    set_env_variable_from_string(args.test)
