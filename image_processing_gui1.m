function image_processing_platform()
    % 创建主界面
    fig = uifigure('Name', '图像处理平台', 'Position', [100, 100, 1200, 800]);

    % 状态指示灯（左上角）
    statusLamp = uilamp(fig, 'Position', [10, 750, 20, 20], 'Color', 'red');
    entropyLabel = uilabel(fig, 'Text', '熵值: --', 'Position', [400, 740, 200, 30]);

    % 创建菜单栏
    menu = uimenu(fig, 'Text', '文件');
    uimenu(menu, 'Text', '打开', 'MenuSelectedFcn', @(src, event) openImage());
    uimenu(menu, 'Text', '保存', 'MenuSelectedFcn', @(src, event) saveImage());

    % 创建选项菜单
    optionsMenu = uimenu(fig, 'Text', '选项');
    uimenu(optionsMenu, 'Text', '重做', 'MenuSelectedFcn', @(src, event) resetImage());

    % 图像显示区（原图像、处理后的图像、RGB折线图并排）
    ax1 = axes(fig, 'Position', [0.05, 0.55, 0.25, 0.35]); % 原图像
    title(ax1, '原图像');

    ax2 = axes(fig, 'Position', [0.4, 0.55, 0.25, 0.35]); % 处理后的图像
    title(ax2, '处理后的图像');

    ax3 = axes(fig, 'Position', [0.75, 0.55, 0.25, 0.35]); % RGB折线图
    title(ax3, 'RGB频度');

  % 图像操作按钮区
btnPanel = uipanel(fig, 'Title', '图像处理操作', 'Position', [0.05, 0.05, 0.9, 0.4]);

% RGB调整控制面板
uilabel(fig, 'Text', '红色通道', 'Position', [880, 280, 80, 25]);
redSlider = uislider(fig, 'Position', [970, 280, 200, 3], 'Limits', [0, 2], 'Value', 1);
redSlider.ValueChangedFcn = @(src, event) adjustRGB();

uilabel(fig, 'Text', '绿色通道', 'Position', [880, 230, 80, 25]);
greenSlider = uislider(fig, 'Position', [970, 230, 200, 3], 'Limits', [0, 2], 'Value', 1);
greenSlider.ValueChangedFcn = @(src, event) adjustRGB();

uilabel(fig, 'Text', '蓝色通道', 'Position', [880, 180, 80, 25]);
blueSlider = uislider(fig, 'Position', [970, 180, 200, 3], 'Limits', [0, 2], 'Value', 1);
blueSlider.ValueChangedFcn = @(src, event) adjustRGB();

uilabel(fig, 'Text', '旋转角度', 'Position', [880, 120, 80, 25]);
rotateAngleEdit = uieditfield(fig, 'numeric', 'Position', [970, 120, 80, 25], 'Value', 0);
rotateAngleEdit.ValueChangedFcn = @(src, event) rotateImage();

 % 噪声添加控制区
    uilabel(fig, 'Text', '均值', 'Position', [180, 255, 80, 25]);%880, 60, 80, 25
    noiseMeanEdit = uieditfield(fig, 'numeric', 'Position', [210, 255, 30, 25], 'Value', 0);


 % 直方图按钮
    histogramButton = uibutton(fig, 'push', 'Text', '直方图', 'Position', [50, 350, 120, 40], ...
                               'ButtonPushedFcn', @(btn, event) showHistogram());
      % 增加灰度化和对比度增强按钮
    contrastEnhanceButton = uibutton(fig, 'push', 'Text', '灰度图', 'Position', [200, 350, 120, 40], ...
                                     'ButtonPushedFcn', @(btn, event) enhanceContrast());
      % 缩放变换按钮
    zoomButton = uibutton(fig, 'push', 'Text', '缩放变换', 'Position', [50, 300, 120, 40], ...
                          'ButtonPushedFcn', @(btn, event) zoomImage());
        % 噪声按钮
    gaussianNoiseButton = uibutton(fig, 'push', 'Text', '高斯噪声', 'Position', [50, 250, 120, 40], ...
                                   'ButtonPushedFcn', @(btn, event) addGaussianNoise());
    saltAndPepperButton = uibutton(fig, 'push', 'Text', '椒盐噪声', 'Position', [250, 250, 120, 40], ...
                                   'ButtonPushedFcn', @(btn, event) addSaltAndPepperNoise());



% 调用 drawnow 强制更新图形
drawnow;

% 调试输出
disp(fig.Children);  % 查看窗口的所有控件
disp(btnPanel.Children);  % 查看面板内的控件
disp(btnPanel.Position);
disp(redSlider.Position);
disp(greenSlider.Position);
disp(blueSlider.Position);
disp(rotateAngleEdit.Position);

    % 初始化图像变量
    img = [];
    processedImg = [];

    % 打开图像函数
    function openImage()
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp', '所有图像文件'}, '选择图像');
        if file
            img = imread(fullfile(path, file));
            % 显示原图像
            imshow(img, 'Parent', ax1);
            
            % 手动计算图像的熵值
            grayImg = rgb2grayCustom(img); % 转换为灰度图像
            entropyValue = calculateEntropy(grayImg); % 计算熵值
            entropyLabel.Text = ['熵值: ', num2str(entropyValue)];
            
            % 计算并显示RGB通道的灰度频度图
            plotRGBHistogram(img);
            
            % 更新状态指示灯为绿色表示成功加载
            statusLamp.Color = 'green';
        end
    end

    % 手动实现RGB到灰度图像的转换
    function grayImg = rgb2grayCustom(img)
        % 获取图像的尺寸
        [height, width, ~] = size(img);
        
        % 初始化灰度图像
        grayImg = zeros(height, width);
        
        % 遍历每个像素进行加权平均
        for i = 1:height
            for j = 1:width
                R = img(i, j, 1);
                G = img(i, j, 2);
                B = img(i, j, 3);
                % 使用加权公式进行转换
                grayImg(i, j) = 0.2989 * R + 0.5870 * G + 0.1140 * B;
            end
        end
        
        % 转换为uint8格式
        grayImg = uint8(grayImg);
    end

    % 手动计算图像的熵值
    function entropyValue = calculateEntropy(grayImg)
        % 计算灰度图像的直方图
        [counts, ~] = imhist(grayImg);
        
        % 归一化直方图，得到每个灰度级的概率
        totalPixels = numel(grayImg);
        probabilities = counts / totalPixels;
        
        % 移除概率为零的项
        probabilities = probabilities(probabilities > 0);
        
        % 使用香农熵公式计算熵值
        entropyValue = -sum(probabilities .* log2(probabilities));
    end

    % 绘制RGB通道的灰度频度图
    function plotRGBHistogram(image)
        % 分离RGB通道
        redChannel = image(:,:,1);
        greenChannel = image(:,:,2);
        blueChannel = image(:,:,3);
        
        % 计算每个通道的直方图
        [countsR, binsR] = imhist(redChannel);
        [countsG, binsG] = imhist(greenChannel);
        [countsB, binsB] = imhist(blueChannel);
        
        % 绘制折线图
        hold(ax3, 'off');
        plot(ax3, binsR, countsR, 'r', 'LineWidth', 2); hold(ax3, 'on');
        plot(ax3, binsG, countsG, 'g', 'LineWidth', 2);
        plot(ax3, binsB, countsB, 'b', 'LineWidth', 2);
        xlabel(ax3, '灰度级');
        ylabel(ax3, '像素数');
        legend(ax3, '红色通道', '绿色通道', '蓝色通道');
        hold(ax3, 'off');
    end

    % 保存图像函数
    function saveImage()
        if isempty(processedImg)
            msgbox('请先处理图像', '错误', 'error');
            return;
        end
        [file, path] = uiputfile({'*.jpg;*.png;*.bmp', '图像文件'}, '保存图像');
        if file
            imwrite(processedImg, fullfile(path, file));
            msgbox('图像已保存', '提示', 'help');
        end
    end

    % 重做函数
    function resetImage()
        if isempty(img)
            msgbox('没有加载图像', '错误', 'error');
            return;
        end
        % 恢复到原始图像
        processedImg = img;
        imshow(img, 'Parent', ax2);
        % 更新RGB三通道的灰度频度图
        plotRGBHistogram(img);
        statusLamp.Color = 'yellow'; % 状态指示灯设置为黄色，表示可以操作
    end

    % 调整RGB通道函数
    function adjustRGB()
        if isempty(img)
            return;
        end
        % 获取当前RGB通道的调整比例
        redFactor = redSlider.Value;
        greenFactor = greenSlider.Value;
        blueFactor = blueSlider.Value;
        
        % 调整RGB通道
        adjustedImg = img;
        adjustedImg(:,:,1) = uint8(double(img(:,:,1)) * redFactor);
        adjustedImg(:,:,2) = uint8(double(img(:,:,2)) * greenFactor);
        adjustedImg(:,:,3) = uint8(double(img(:,:,3)) * blueFactor);
        
        % 显示调整后的图像
        processedImg = adjustedImg;
        imshow(adjustedImg, 'Parent', ax2);
        plotRGBHistogram(adjustedImg);  % 更新RGB频度图
    end
 % 旋转图像函数
    function rotateImage()
        if isempty(img)
            return;
        end
        % 获取旋转角度
        angle = rotateAngleEdit.Value;
        
        % 使用imrotate函数旋转图像
        rotatedImg = imrotate(img, angle);
        
        % 显示旋转后的图像
        processedImg = rotatedImg;
        imshow(rotatedImg, 'Parent', ax2);
        plotRGBHistogram(rotatedImg);  % 更新RGB频度图
    end

    function showHistogram()
    % 弹出新窗口显示灰度直方图、均衡化后图像、匹配后图像
    histFig = uifigure('Name', '直方图处理', 'Position', [200, 200, 1000, 600]);

    % 创建三个子图区域，确保它们有足够的空间
    ax1 = axes(histFig, 'Position', [0.05, 0.3, 0.3, 0.6]); % 灰度直方图
    title(ax1, '灰度直方图', 'FontSize', 12);
    
    ax2 = axes(histFig, 'Position', [0.4, 0.3, 0.3, 0.6]); % 直方图均衡化
    title(ax2, '直方图均衡化', 'FontSize', 12);
    
    ax3 = axes(histFig, 'Position', [0.75, 0.3, 0.3, 0.6]); % 直方图匹配（规定化）
    title(ax3, '直方图匹配', 'FontSize', 12);

    % 处理图像
    if isempty(img)
        return;
    end
    
    % 转换为灰度图像
    grayImg = rgb2gray(img);
    
    % 绘制灰度直方图
    histogramData = computeHistogram(grayImg);
    bar(ax1, histogramData, 'BarWidth', 1);
    
    % 直方图均衡化
    eqImg = histogramEqualization(grayImg);
    imshow(eqImg, 'Parent', ax2);
    
    % 直方图匹配（规定化）
    %targetImg = zeros(size(grayImg), 'like', grayImg); % 可以设置目标图像
    targetImg = imread('C:\Users\10835\Desktop\beijing.jpg'); 
    if size(targetImg, 3) == 3
    targetImg = rgb2gray(targetImg);
    end
    matchedImg = histogramMatching(grayImg, targetImg);
    imshow(matchedImg, 'Parent', ax3);
end

    % 计算灰度直方图
    function histData = computeHistogram(grayImage)
        histData = zeros(1, 256);
        for i = 1:numel(grayImage)
            histData(grayImage(i) + 1) = histData(grayImage(i) + 1) + 1;
        end
    end
    
    % 直方图均衡化
    function eqImage = histogramEqualization(grayImage)
        % 计算累积分布函数 (CDF)
        histData = computeHistogram(grayImage);
        cdf = cumsum(histData) / numel(grayImage);
        % 归一化映射
        eqImage = uint8(255 * cdf(double(grayImage) + 1));
    end
    
    % 直方图匹配
    function matchedImage = histogramMatching(sourceImg, targetImg)
    % 计算源图像和目标图像的灰度直方图
    sourceHist = computeHistogram(sourceImg);
    targetHist = computeHistogram(targetImg);
    
    % 计算源图像和目标图像的累积分布函数 (CDF)
    sourceCDF = cumsum(sourceHist) / numel(sourceImg);
    targetCDF = cumsum(targetHist) / numel(targetImg);
    
    % 创建一个空的匹配图像
    matchedImage = zeros(size(sourceImg), 'like', sourceImg);
    
    % 直方图匹配过程
    for i = 1:numel(sourceImg)
        % 找到最接近的目标CDF值，并映射到目标像素值
        [~, idx] = min(abs(sourceCDF(sourceImg(i) + 1) - targetCDF));
        
        % 确保映射后的像素值位于有效范围内
        matchedImage(i) = idx - 1;  % -1 是因为索引从0开始
    end
    
    % 确保匹配后的图像值在[0, 255]范围内
    matchedImage = uint8(matchedImage);
    
    % 显示一些调试信息
    fprintf('源图像灰度最小值: %d, 最大值: %d\n', min(sourceImg(:)), max(sourceImg(:)));
    fprintf('目标图像灰度最小值: %d, 最大值: %d\n', min(targetImg(:)), max(targetImg(:)));
    fprintf('匹配图像灰度最小值: %d, 最大值: %d\n', min(matchedImage(:)), max(matchedImage(:)));
    end

% 对比度增强
function enhanceContrast()
    if isempty(img)
        return;
    end
    
    % 灰度化图像
    grayImg = rgb2grayCustom(img);

    % 创建一个弹出窗口，显示四张图像（2x2布局）
    figure('Name', '图像处理结果', 'Position', [200, 200, 1200, 800]);

    % 第一个子图：灰度图
    subplot(2, 2, 1);
    imshow(grayImg);
    title('灰度图');

    % 第二个子图：线性对比度增强
    linearEnhanced = imadjust(grayImg, [0.2, 0.8], [0, 1]);
    subplot(2, 2, 2);
    imshow(linearEnhanced);
    title('线性对比度增强');

    % 第三个子图：对数变换增强
    grayImgDouble = double(grayImg);  % 将灰度图像转换为 double 类型
    c = 255 / log(1 + max(grayImgDouble(:))); % 缩放常数
    logEnhanced = c * log(1 + grayImgDouble); % 对数变换
    subplot(2, 2, 3);
    imshow(uint8(logEnhanced));  % 显示图像时转换回 uint8 类型
    title('对数变换增强');

    % 第四个子图：指数变换增强
    gamma = 1.5;  % 可以调整gamma值来控制指数变换的强度
    expEnhanced = 255 * (grayImgDouble / 255) .^ gamma;  % 归一化到[0, 1]范围后进行指数变换
    subplot(2, 2, 4);
    imshow(uint8(expEnhanced));
    title('指数变换增强');
end

  % 缩放变换函数
    function zoomImage()
        if isempty(img)
            return;
        end

        % 获取缩放比例，可以通过UI控件来修改
        scale = 1.2;  % 例如缩放1.2倍

        % 计算新的图像大小
        [rows, cols, channels] = size(img);
        newRows = round(rows * scale);
        newCols = round(cols * scale);

        % 创建一个新的空图像
        zoomedImg = zeros(newRows, newCols, channels, 'uint8');

        % 使用最近邻插值进行缩放
        for r = 1:newRows
            for c = 1:newCols
                oldR = round(r / scale);
                oldC = round(c / scale);
                oldR = min(max(oldR, 1), rows);  % 防止超出范围
                oldC = min(max(oldC, 1), cols);  % 防止超出范围
                zoomedImg(r, c, :) = img(oldR, oldC, :);
            end
        end

        % 显示缩放后的图像
        processedImg = zoomedImg;
        imshow(processedImg, 'Parent', ax2);
    end

% 高斯噪声添加函数
function addGaussianNoise()
    if ~isempty(img)
        meanNoise = noiseMeanEdit.Value;  % 获取用户输入的噪声均值
        sigma = 0.1;  % 设置标准差
        noisyImg = double(img) + meanNoise + sigma * randn(size(img));  % 转换为double进行加法操作
        noisyImg = uint8(noisyImg);  % 强制转换为uint8，确保图像像素为有效值范围
        
        % 防止像素值超出0-255范围
        noisyImg(noisyImg > 255) = 255;  
        noisyImg(noisyImg < 0) = 0;
        
        % 显示噪声图像
        imshow(noisyImg, 'Parent', ax2);
        processedImg = noisyImg;
        disp('添加高斯噪声');
    end
end

% 椒盐噪声添加函数
function addSaltAndPepperNoise()
    if ~isempty(img)
        meanNoise = noiseMeanEdit.Value;  % 获取用户输入的噪声均值
        noisyImg = img;
        p = 0.01;  % 控制椒盐噪声的概率（噪声强度），0.01 代表1%的像素
        numSalt = round(p * numel(img));  % 计算盐噪声的数量
        numPepper = numSalt;  % 胡椒噪声数量与盐噪声相同

        % 随机生成盐噪声（白色）
        saltIndices = randperm(numel(img), numSalt);
        noisyImg(saltIndices) = 255;

        % 随机生成胡椒噪声（黑色）
        pepperIndices = randperm(numel(img), numPepper);
        noisyImg(pepperIndices) = 0;

        % 显示带噪声图像
        imshow(noisyImg, 'Parent', ax2);
        processedImg = noisyImg;
        disp('添加椒盐噪声');
    end
end

end


