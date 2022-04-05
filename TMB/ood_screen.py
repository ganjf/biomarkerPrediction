# @author Jiefeng Gan
import argparse
import pandas as pd


def OOD_distinguish(inputCSV, outputCSV, conf_label='confidence', threshold=0.5):
    data = pd.read_csv(inputCSV)
    raw_pid_lst = list(data['name'])
    raw_pid_lst = list(set([x[0:12] for x in raw_pid_lst])) # TCGA
    # raw_pid_lst = list(set([x.split('.')[0][:-3] for x in raw_pid_lst])) # CPTAC
    print('Raw patient num: ', len(raw_pid_lst))
    data_clean = data[data[conf_label] >= threshold]
    iod_pid_lst = list(data_clean['name'])
    iod_pid_lst = list(set([x[0:12] for x in iod_pid_lst])) # TCGA
    # iod_pid_lst = list(set([x.split('.')[0][:-3] for x in iod_pid_lst])) # CPTAC
    print('IOD patient num: ', len(iod_pid_lst))
    if len(raw_pid_lst) == len(iod_pid_lst):
        data_clean.to_csv(outputCSV, index=False)
    else:
        pid_lst = list(data['name'])
        pid_lst = [x[0:12] for x in pid_lst] # TCGA
        # pid_lst = [x.split('.')[0][:-3] for x in pid_lst] # CPTAC
        data['pid'] = pid_lst
        data_clean = data[data[conf_label] >= threshold]
        lost_pid = list(set(raw_pid_lst) - set(iod_pid_lst))
        for pid in lost_pid:
            lost_data = data[data['pid'] == pid]
            lost_data = lost_data.sort_values(by=conf_label, ascending=False)
            index_lst = lost_data.index.to_list()
            num = min(20, len(lost_data))
            for i in range(num):
                data_clean = data_clean.append(data.iloc[index_lst[i]], ignore_index=True)
        data_clean = data_clean.sample(len(data_clean))
        data_clean.drop(labels='pid', axis=1, inplace=True)
        iod_pid_lst = list(data_clean['name'])
        iod_pid_lst = list(set([x[0:12] for x in iod_pid_lst])) # TCGA
        # iod_pid_lst = list(set([x.split('.')[0][:-3] for x in iod_pid_lst])) # CPTAC
        print('IOD patient num after pid supplementary: ', len(iod_pid_lst))
        data_clean.to_csv(outputCSV, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='+')
    parser.add_argument('--output', type=str, nargs='+')
    parser.add_argument('--confidence_threshold', type=float, nargs='+')
    args = parser.parse_args()

    input_output = zip(args.input, args.output, args.confidence_threshold)

    print(args.input)
    print(args.output)
    print(args.confidence_threshold)
    # exit()

    for in_csv, out_csv, threshold in input_output:
        threshold /= 100.0
        print('confidence score >= {}'.format(threshold))
        print('input: {}'.format(in_csv))
        print('Save to: {}'.format(out_csv))
        OOD_distinguish(
            inputCSV=in_csv,
            outputCSV=out_csv,
            threshold=threshold
        )