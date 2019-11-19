% BM3D takes [0:1] image
slice = @my_bm3d;
slice_auto = @bm3d_auto;

raw = {noisy_fixed_135 noisy_fixed_255 noisy_fixed_623 noisy_fixed_653 noisy_fixed_682 noisy_fixed_827 noisy_fixed_967};

slice_5_g2 = slice_auto(raw, 5, 7);
save("slice_5_g2.mat", "slice_5_g2");

function y = bm3d_auto(raw, sigma, num)
    y = cell(1,num);
    for i = 1:num
        sample_min = min(raw{i}, [], 'all');
        sample_max = max(raw{i}, [], 'all');
        normalized_input = (raw{i}(:,:,:)-sample_min)/(sample_max-sample_min);
        denoised_result = zeros(64, 256, 256);
        for j = 1:64
            [psnr, denoised_result(j,:,:)] = BM3D(1, squeeze(normalized_input(j,:,:)), sigma);
        end
        sprintf('# %d input done', i)
        y{i} = denoised_result*(sample_max-sample_min) + sample_min;
    end
end

function y = my_bm3d(x, sigma)
    raw_min = min(x, [], 'all');
    raw_max = max(x, [], 'all');
    normalized = (x(:,:,:)-raw_min)/(raw_max-raw_min);
    
    for i = 1:64
        [psnr, denoised(i,:,:)] = BM3D(1, squeeze(normalized(i,:,:)), sigma);
    end
    
    y = denoised*(raw_max-raw_min) + raw_min;
end
