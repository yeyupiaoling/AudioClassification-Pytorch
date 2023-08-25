#!/bin/bash

download_dir=dataset/language


[ ! -d ${download_dir} ] && mkdir -p ${download_dir}

if [ ! -f ${download_dir}/test.tar.gz ]; then
    echo "准备下载测试集"
    wget --no-check-certificate https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/3D-Speaker/test.tar.gz -P ${download_dir}
    md5=$(md5sum ${download_dir}/test.tar.gz | awk '{print $1}')
    [ $md5 != "45972606dd10d3f7c1c31f27acdfbed7" ] && echo "Wrong md5sum of 3dspeaker test.tar.gz" && exit 1
fi

if [ ! -f ${download_dir}/train.tar.gz ]; then
    echo "准备下载训练集"
    wget --no-check-certificate https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/3D-Speaker/train.tar.gz -P ${download_dir}
    md5=$(md5sum ${download_dir}/train.tar.gz | awk '{print $1}')
    [ $md5 != "c2cea55fd22a2b867d295fb35a2d3340" ] && echo "Wrong md5sum of 3dspeaker train.tar.gz" && exit 1
fi

echo "下载完成！"

echo "准备解压"

tar -zxvf ${download_dir}/train.tar.gz -C ${rawdata_dir}/
tar -xzvf ${download_dir}/test.tar.gz -C ${rawdata_dir}/

echo "解压完成！"
