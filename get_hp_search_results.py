import argparse, os

def get_auprcs_from_log(log_path, auprc_string="AUPRC:"):
    auprcs = []
    with open(log_path) as f:
        for line in f:
            if auprc_string in line:
                auprcs.append(float(line.split()[-1]))
    return auprcs

def get_best_auprc(log_path, auprc_string="AUPRC:"):
    auprcs = get_auprcs_from_log(log_path, auprc_string)
    try:
        return max(auprcs)
    except:
        return 0

def get_best_auprcs(parent_dname, auprc_string="AUPRC:"):
    best_auprcs = {}
    for dname in os.listdir(parent_dname):
        best_auprcs[dname] = get_best_auprc(os.path.join(parent_dname, dname, "log.out"), auprc_string)
    return best_auprcs

def main():
    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    with open(args.output_path, 'w') as f:
        for parent_dname in args.parent_dnames:
            best_auprcs = get_best_auprcs(parent_dname, "val AUPRC:")
            f.write("Parent directory: " + parent_dname + "\n")
            for dname, auprc in best_auprcs.items():
                f.write("{}: {}\n".format(dname, auprc))
            best_hp_config_dname = max(best_auprcs, key=lambda key:best_auprcs[key])
            best_auprc = best_auprcs[best_hp_config_dname]
            f.write("***Best validation auprc is {}, corresponding to experiment {}\n".format(
                best_auprc, best_hp_config_dname)
            )
            f.write("-------------\n")

parser = argparse.ArgumentParser()
parser.add_argument("--parent-dnames", nargs="+")
parser.add_argument("--output-path")

if __name__ == "__main__":
    main()
