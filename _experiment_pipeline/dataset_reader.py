import os


def get_data_samples_dictionary(tail):
    data_samples = {}

    root = os.path.join(tail, '_experiment_pipeline', 'data_sets')
    for file in os.listdir(root):
        data_samples[file] = {}

    for key in data_samples.keys():
        curr_path = os.path.join(root, key)
        for data_set in os.listdir(curr_path):
            data_samples[key][data_set] = {"grid": [], "samples": []}
            sample_root = os.path.join(curr_path, data_set)
            for sample_name in os.listdir(sample_root):
                if "sample_" in sample_name:
                    data_samples[key][data_set]["samples"].append(os.path.join(sample_root, sample_name))
                elif "grid" in sample_name:
                    data_samples[key][data_set]["grid"].append(os.path.join(sample_root, sample_name))
                elif "ground_truth" in sample_name:
                    data_samples[key][data_set]["ground_truth"] = os.path.join(sample_root, sample_name)
                else:
                    print("unknown data sample naming: " + sample_name)

    return data_samples
