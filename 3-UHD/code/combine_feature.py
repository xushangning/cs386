import numpy as np
import os


def load_features(fname, mode='train'):
    data = {}
    if mode in ['train', 'all']:
        data['4k'] = np.loadtxt('images/4k/' + fname)
        assert np.isnan(data['4k']).sum() == 0, "Missing data"
        data['1080bl'] = np.loadtxt('images/1080P/bilinear/' + fname)
        assert np.isnan(data['1080bl']).sum() == 0, "Missing data"
        data['1080bc'] = np.loadtxt('images/1080P/bicubic/' + fname)
        assert np.isnan(data['1080bc']).sum() == 0, "Missing data"
        data['1080lz'] = np.loadtxt('images/1080P/lanczos2/' + fname)
        assert np.isnan(data['1080lz']).sum() == 0, "Missing data"
        data['720bl'] = np.loadtxt('images/720P/bilinear/' + fname)
        assert np.isnan(data['720bl']).sum() == 0, "Missing data"
        data['720bc'] = np.loadtxt('images/720P/bicubic/' + fname)
        assert np.isnan(data['720bc']).sum() == 0, "Missing data"
        data['720lz'] = np.loadtxt('images/720P/lanczos2/' + fname)
        assert np.isnan(data['720lz']).sum() == 0, "Missing data"
    elif mode in ['test', 'all']:
        data['1080sr'] = np.loadtxt('images/1080P/srcnn/' + fname)
        assert np.isnan(data['1080sr']).sum() == 0, "Missing data"
        data['1080ed'] = np.loadtxt('images/1080P/edsr/' + fname)
        assert np.isnan(data['1080ed']).sum() == 0, "Missing data"
        data['720sr'] = np.loadtxt('images/720P/srcnn/' + fname)
        assert np.isnan(data['720sr']).sum() == 0, "Missing data"
        data['720ed'] = np.loadtxt('images/720P/edsr/' + fname)
        assert np.isnan(data['720ed']).sum() == 0, "Missing data"
    return data


def concate_features(fname_list, prefix='all', mode='train'):
    data = None
    for fname in fname_list:
        data_ = load_features(fname, mode=mode)
        if data is None:
            data = data_
        else:
            for key, item in data.items():
                data[key] = np.concatenate([item, data_[key]], axis=1)
    false_data = None
    for key, item in data.items():
        if key == '4k':
            continue
        if false_data is None:
            false_data = item[:, :]
        else:
            false_data = np.concatenate([false_data, item], axis=0)
    if '4k' in data.keys():
        np.savetxt('images/' + prefix + '_true.txt', data['4k'])
    np.savetxt('images/' + prefix + '_false.txt', false_data)
    # print(false_data)


if __name__ == "__main__":
    fname_list = [
        # 'tile_32_channel_0_samples_50_rate_2_threshold_20_div_True_offset_1.0.txt',
        'tile_32_channel_0_samples_50_rate_2_threshold_20_div_True_offset_1.0_ref_AR.txt',
        'tile_32_channel_0_samples_50_rate_2_threshold_20_div_True_offset_1.0_ref_NN.txt',
        'tile_32_channel_0_samples_50_rate_2_threshold_20_div_True_offset_1.0_ref_BL.txt',
        'tile_32_channel_0_samples_50_rate_2_threshold_20_div_True_offset_1.0_ref_BC.txt',
        'tile_32_channel_0_samples_50_rate_2_threshold_20_div_True_offset_1.0_ref_NO.txt',
        'tile_32_channel_0_samples_50_rate_3_threshold_20_div_True_offset_1.0_ref_AR.txt',
    ]
    # concate_features(fname_list, prefix='all', mode='train')
    concate_features(fname_list, prefix='test', mode='test')
