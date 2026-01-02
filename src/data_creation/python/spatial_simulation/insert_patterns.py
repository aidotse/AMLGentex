import sys
import yaml
import os


def insert_normal_patterns(conf_file:str):
    pass


def insert_alert_patterns(conf_file:str):

    with open(conf_file, 'r') as rf:
        conf = yaml.safe_load(rf)
        directory = conf["input"]["directory"]
        output_dir = conf["temporal"]["directory"]
        patterns_to_insert = conf["input"]["insert_patterns"]

    src_path = os.path.join(directory, patterns_to_insert)
    dst_path = os.path.join(output_dir, 'alert_members.csv')
    
    with open(src_path, 'r') as src_file, open(dst_path, 'a') as dst_file:
        next(src_file)
        accounts = []
        for line in src_file:
            accounts.append(line.split(',')[2])
            dst_file.write(line)
    
    modify_accounts_file(conf_file, accounts)


def modify_accounts_file(conf_file:str, accounts:list):

    with open(conf_file, 'r') as rf:
        conf = yaml.safe_load(rf)
        output_dir = conf["temporal"]["directory"]

    accounts_file = os.path.join(output_dir, 'accounts.csv')
    accounts = sorted(accounts)
    lines = []
    # find the accounts in the accounts file and write is_sar
    with open(accounts_file, 'r') as rf:
        i = 0
        for line in rf:
            acct = line.split(',')[0]
            if acct==accounts[i]:
                # replace false with true
                lines.append(line.replace('false', 'true'))
                i+=1
            else:
                lines.append(line)
    with open(accounts_file, 'w') as wf:
        for line in lines:
            wf.write(line)
    

def main():
    """Main entry point for inserting patterns"""
    argv = sys.argv

    if len(argv) == 1:
        # debug
        PARAM_FILES = '100K_accts_inserted_alerts'
        conf_file = f'paramFiles/{PARAM_FILES}/conf.json'
        insert_alert_patterns(conf_file)
    elif len(argv) == 2:
        conf_file = argv[1]
        insert_alert_patterns(conf_file)
    elif len(argv) == 3:
        conf_file = argv[1]
        type = argv[2]
        if type == 'normal':
            insert_normal_patterns(conf_file)
        elif type == 'alert':
            insert_alert_patterns(conf_file)
    else:
        print("Usage: python insert_patterns.py <config_file> [<type>]")
        print("config_file: path to the configuration file")
        print("type (optional): type of pattern to insert: 'normal' or 'alert'. Default is 'alert'")
        sys.exit(1)


if __name__ == "__main__":
    main()
    