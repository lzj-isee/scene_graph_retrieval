### 1. 对VG数据集测试集中所有图片生成场景图  

##### 该步骤作用  

对VG数据集的测试集中的所有图片生成场景图，输出两个文件`data_info.json`（保存图片路径和标签信息）和`filted_scene_graph.pkl`（生成的场景图，以dict形式存放）

##### 运行方式  

运行`gen_sg_VGtest.sh`文件  

##### 参数说明（参考项目https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch）  

* `CUDA_VISIBLE_DEVICES`: 可用的显卡ID
* `nproc_per_node`: 选用的显卡数量（进程数）
* `TEST.IMS_PER_BATCH`: 单显卡每batch负载图片数量，2080ti用4
* `DTYPE`: 浮点模式（单精度、半精度）
* `GLOVE_DIR`: Word Embedding 要用到的模型，该模型存储的文件夹（load and save），第一次用会自动下载
* `MODEL.PRETRAINED_DETECTOR_CKPT`: 模型存储的文件夹（load）
* `OUTPUT_DIR`: 模型存储的文件夹（load），注意该文件夹下有个`last_checkpoint`文件，修改里面的路径为同文件夹下的`model_0028000.pth`的绝对路径
* `TEST.CUSTUM_EVAL`: 是否开启CUSTUM模式，这里是False
* `DETECTED_SGG_DIR`: Scene Graph输出的文件夹（save），在该文件夹下保存`data_info.json`和`filted_scene_graph.pkl`文件

##### 其他说明

`Image2SceneGraph`文件夹下的`sgg_post_process.py`文件内的`custom_sgg_post_precessing_and_save`函数定义了scene graph的过滤规则以及存储规则，有需求可改

### 2. 生成测试集  

##### 该步骤作用

对`Dataset/query_text.json`文件中的query场景，从VG测试集中检索出符合场景的图片，输出图片的场景标签文件`categorize_info_raw.pkl`，以及将检索的图片分类别存放至`--out_image_dir`文件夹  

##### 运行方式  

运行`categorize_images.sh`文件  

##### 参数说明  

* `query_text_path`: 参与生成数据集的场景（load）
* `scene_graphs_path`: 场景图（load）
* `data_info_path`: 图片路径信息和标签信息文件（load）
* `src_image_dir`: 完整VG数据集的图片路径（load）
* `out_image_dir`: 存放输出图片的路径（save）
* `out_info_dir`: 存放打标信息路径（save）

### 3. 检索及评测  

##### 该步骤作用  

基于步骤1生成的场景图以及步骤2生成的数据集，生成检索结果以及评测precision和recall指标。输出三个文件：`ranking_result.pkl`存放未过滤的检索结果；`ranking_result_filted.pkl`存放过滤的检索结果；`evaluation_result.pkl`存放评测结果。

##### 运行方式  

运行`gen_ranking_result.sh`文件  

##### 参数说明  

* `scene_graphs_path`: 场景图（load）
* `categorize_info_path`: 场景ground truth（load）
* `out_dir`: 结果输出文件夹（save）

### *可视化生成的场景图  

##### 运行方式  

运行`visualize_costom_image.ipynb`文件  

### *可视化检索结果  

##### 该步骤作用  

建立所有query项的检索结果，包含TP、FP、FN三项归类  

##### 运行方式  

运行`Evaluation`文件夹下的`ranking_check.py`文件  

##### 参数说明  

* `ranking_result_path`: 检索结果路径（load）
* `cat_info_path`: 场景ground truth（load）
* `src_image_dir`: 完整VG数据集的图片路径（load）
* `out_dir`: 结果输出文件夹（save）

<!-- ### 1. 根据AAAI 2020文章“Image-to-Image Retrieval by Learning Similarity between Scene Graphs”，从VisualGenome数据集中抽取有标签的一分部图片作为query集（共3216张图片）

##### 该步骤作用

从完整的VG数据集中抽取一部分图片作为query集合，存储在`/query`文件夹下

##### 运行方式

运行`Dataset`文件夹下的`gen_test_dataset.py`文件

##### 参数说明

* `image_folder`: 完整的存放VG数据集图片的源文件夹（load）
* `target_folder`: 存放query图片的目标文件夹（save）
* `annotation_results_path`: AAAI2020论文的打标结果（load）
* `triplet_information_path`: AAAI2020论文的打标索引（load）

### 2. 存储相关性结果以及检查query数据集的相关性打标的正确性

##### 该步骤作用

1. 对于query中的每张图片，建立与其相关的图片索引集合，该索引集合存储在`/outputs`文件夹下，输出两个文件`relevance_single.pkl`和`relevance_multi.pkl`
2. 对于query中的某张图片，可视化与其相关的每张图片，结果不存储，显示在jupyter中

##### 运行方式

运行`Dataset`文件夹下的`check_test_dataset.ipynb`文件

##### 参数说明

* `image_folder`: 完整的存放VG数据集图片的源文件夹（load）
* `annotation_results_path`: AAAI2020论文的打标结果（load）
* `triplet_information_path`: AAAI2020论文的打标索引（load）
* `out_folder`: 存储索引集合的文件夹（save）
* `relevance_thr`: 阈值，对于某对图片，若（正投票-负投票）大于该阈值，则认为建立了相关性
* `multi_relation`: 是否考虑连接关系：我相关图片的相关图片依然是我的相关图片，若为False，则不输出`relevance_multi.pkl`文件
* `checking_id`: 被检查的图片的id（文件名，不含后缀）

### 3. 为每张query图片生成对应的场景图

##### 该步骤作用

对`/query`文件夹下的每张图片，生成对应的场景图，结果存放在`/outputs`文件夹下的`filted_scene_graph.pkl`文件

##### 运行方式

运行`Image2SceneGraph`文件夹下的`gen_scene_graph.sh`文件

##### 参数说明（没提到的参数建议用默认的，参考项目 https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch）

* `CUDA_VISIBLE_DEVICES`: 可用的显卡ID
* `nproc_per_node`: 选用的显卡数量（进程数）
* `TEST.IMS_PER_BATCH`: 单显卡每batch负载图片数量，2080ti用4
* `DTYPE`: 浮点模式（单精度、半精度）
* `GLOVE_DIR`: Word Embedding 要用到的模型，该模型存储的文件夹（load and save），第一次用会自动下载
* `MODEL.PRETRAINED_DETECTOR_CKPT`: 模型存储的文件夹（load）
* `OUTPUT_DIR`: 模型存储的文件夹（load），注意该文件夹下有个`last_checkpoint`文件，修改里面的路径为同文件夹下的`model_0028000.pth`的绝对路径
* `TEST.CUSTUM_EVAL`: 是否开启CUSTUM模式
* `TEST.CUSTUM_PATH`: 测试图片的存储文件夹（load），设置为第一步的`\query`文件夹
* `DETECTED_SGG_DIR`: Scene Graph输出的文件夹（save），在该文件夹下保存`filted_scene_graph.pkl`文件

##### 其他说明

`Image2SceneGraph`文件夹下的`sgg_post_process.py`文件内的`custom_sgg_post_precessing_and_save`函数定义了scene graph的过滤规则以及存储规则，有需求可改

### 4. 对每张query图片，计算基于场景图匹配的相似度

##### 该步骤作用

对`filted_scene_graph.pkl`文件中的所有item，计算其与其他item间的相似度

##### 运行方式

运行`GraphMatching`文件夹下的`gen_ranking_result.sh`文件

##### 参数说明

* `nproc_per_node`: 进程数
* `scene_graph_path`: 场景图存储路径（load）
* `target_folder`: 相关性结果的保存文件夹（save），在该文件夹下保存名为`ranking_results.pkl`的文件

##### 其他说明

代码不调用显卡，用CPU完成运算，参数`nproc_per_node`依据CPU性能和内存设置。

### 5. 可视化基于GM的ranking结果

##### 该步骤作用

对`ranking_results.pkl`中任意的item，可视化基于场景图匹配的ranking结果

##### 运行方式

运行`GraphMatching`文件夹下的`check_ranking_result.ipynb`文件

##### 参数说明

* `ranking_result_path`: 第4步ranking结果的保存路径（load）
* `image_folder`: 读取VG数据集图片的文件夹（load）
* `relevance_thr`: 将相关性低于该阈值的ranking结果过滤掉，取值[0, 1]
* `checking_id`: 被检查图片的id（文件名，不含后缀）

### 6. 指标评测

##### 该步骤作用

计算precision、recall、human_agreement等指标

##### 运行方式

1. 计算precision@k、recall@k指标：运行`Evaluation`文件夹下的`gen_precision_recall.py`文件
2. 计算human_agreement指标: 运行`Evaluation`文件夹下的`gen_human_agreement.py`文件 (参考https://www.aaai.org/AAAI21Papers/AAAI-9214.YoonS.pdf)

##### 参数说明

* `relevance_gt_path`: 步骤2的图片相关性ground_truth，可选文件`relevance_single.pkl`或`relevance_multi.pkl`（load）
* `relevance_pred_path`: 步骤4的ranking_results文件`ranking_results.pkl`（load）
* `annotation_results_path`: AAAI2020论文的打标结果（load）
* `triplet_information_path`: AAAI2020论文的打标索引（load）

##### 其他说明

对于precision@k、recall@k指标，可能会发生prediction中相关的item数目小于k的情况，这里的k取值为$\min(len(prediction), k)$ -->