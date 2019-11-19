sigmaMat = zeros(64, 256, 256, 'double');

sigma = 0.005;

raw = {noisy_fixed_135 noisy_fixed_255 noisy_fixed_623 noisy_fixed_653 noisy_fixed_682 noisy_fixed_827 noisy_fixed_967};

volume_Rice_g2 = bm4d_Rice(raw, sigma, 9);
volume_Gauss_g2 = bm4d_Gauss(raw, sigma, 9);

save("volume_Rice_g2.mat", "volume_Rice_g2");
save("volume_Gauss_g2.mat", "volume_Gauss_g2");

function y = bm4d_Rice(raw, sigma, num)
    y = cell(1,num);
    for i = 1:num
        [y{i}, sigmaMat] = bm4d(raw{i}, 'Rice', sigma);
    end
end

function y = bm4d_Gauss(raw, sigma, num)
    y = cell(1,num);
    for i = 1:num
        [y{i}, sigmaMat] = bm4d(raw{i}, 'Gauss', sigma);
    end
end

% [volume_Gauss_new_20180307, sigmaMat] = bm4d(new_20180307, 'Gauss', sigma);
% [volume_Rice_new_20180307, sigmaMat] = bm4d(new_20180307, 'Rice', sigma);
% [volume_Gauss_new_20181017, sigmaMat] = bm4d(new_20181017, 'Gauss', sigma);
% [volume_Rice_new_20181017, sigmaMat] = bm4d(new_20181017, 'Rice', sigma);
% [volume_Gauss_new_20181122, sigmaMat] = bm4d(new_20181122, 'Gauss', sigma);
% [volume_Rice_new_20181122, sigmaMat] = bm4d(new_20181122, 'Rice', sigma);
% [volume_Gauss_new_20181123, sigmaMat] = bm4d(new_20181123, 'Gauss', sigma);
% [volume_Rice_new_20181123, sigmaMat] = bm4d(new_20181123, 'Rice', sigma);
% [volume_Gauss_new_20190102, sigmaMat] = bm4d(new_20190102, 'Gauss', sigma);
% [volume_Rice_new_20190102, sigmaMat] = bm4d(new_20190102, 'Rice', sigma);
% [volume_Gauss_new_20190307, sigmaMat] = bm4d(new_20190307, 'Gauss', sigma);
% [volume_Rice_new_20190307, sigmaMat] = bm4d(new_20190307, 'Rice', sigma);
% [volume_Gauss_new_20190709, sigmaMat] = bm4d(new_20190709, 'Gauss', sigma);
% [volume_Rice_new_20190709, sigmaMat] = bm4d(new_20190709, 'Rice', sigma);

% [volume_Gauss_20181114, sigmaMat] = bm4d(raw_20181114, 'Gauss', sigma);
% [volume_Rice_20181114, sigmaMat] = bm4d(raw_20181114, 'Rice', sigma);
% [volume_Gauss_20190107, sigmaMat] = bm4d(raw_20190107, 'Gauss', sigma);
% [volume_Rice_20190107, sigmaMat] = bm4d(raw_20190107, 'Rice', sigma);
% [volume_Gauss_20190307, sigmaMat] = bm4d(raw_20190307, 'Gauss', sigma);
% [volume_Rice_20190307, sigmaMat] = bm4d(raw_20190307, 'Rice', sigma);
% [volume_Gauss_20190327, sigmaMat] = bm4d(raw_20190327, 'Gauss', sigma);
% [volume_Rice_20190327, sigmaMat] = bm4d(raw_20190327, 'Rice', sigma);
% [volume_Gauss_20190516, sigmaMat] = bm4d(raw_20190516, 'Gauss', sigma);
% [volume_Rice_20190516, sigmaMat] = bm4d(raw_20190516, 'Rice', sigma);
% [volume_Gauss_20190621, sigmaMat] = bm4d(raw_20190621, 'Gauss', sigma);
% [volume_Rice_20190621, sigmaMat] = bm4d(raw_20190621, 'Rice', sigma);
% [volume_Gauss_20190702, sigmaMat] = bm4d(raw_20190702, 'Gauss', sigma);
% [volume_Rice_20190702, sigmaMat] = bm4d(raw_20190702, 'Rice', sigma);