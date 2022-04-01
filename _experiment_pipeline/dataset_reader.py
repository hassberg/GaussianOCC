import os


def get_data_samples_dictionary(tail):
    data_samples = {"continuous": {}, "discrete": {}}

    root = os.path.join(tail, '_experiment_pipeline', 'data_sets')
    for key in data_samples.keys():
        curr_path = os.path.join(root, key)
        for data_set in os.listdir(curr_path):
            data_samples[key][data_set] = {"test": [], "samples": []}
            sample_root = os.path.join(curr_path, data_set)
            for sample_name in os.listdir(sample_root):
                if "sample_" in sample_name:
                    data_samples[key][data_set]["samples"].append(os.path.join(sample_root, sample_name))
                elif "test" in sample_name:
                    data_samples[key][data_set]["test"].append(os.path.join(sample_root, sample_name))
                else:
                    print("unknown data sample naming: " + sample_name)

    return data_samples
