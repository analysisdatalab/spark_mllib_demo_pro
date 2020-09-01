package com.java.sparkml.ml.textclassification;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import org.wltea.analyzer.app.WordCountAnalyse;

import java.io.IOException;

/**
 * @Author:
 * @Date: 2020-07-24 15:04
 * @Version: 1.0
 * @Modified By:
 * @Description:
 */
/*
从 HDFS 上读取原始数据集，并创建一个 DataFrame。
使用 StringIndexer 将原始的文本标签 (“Ham”或者“Spam”) 转化成数值型的表型，以便 Spark ML 处理。
使用 Word2Vec 将短信文本转化成数值型词向量。
使用 MultilayerPerceptronClassifier 训练一个多层感知器模型。
使用 LabelConverter 将预测结果的数值标签转化成原始的文本标签。
最后在测试数据集上测试模型的预测精确度。

 */

// 参考链接：https://blog.csdn.net/weixin_32265569/article/details/107604269

public class T06_Classification_MultiWays {
    public static void main(String[] args) throws IOException {
        Logger.getLogger("org").setLevel(Level.WARN);
        // 1.初始化spark配置信息并建立与spark的连接
        SparkSession spark = SparkSession.builder().appName("T13_StockNewsClassification_MultiWays").master("local").getOrCreate();

        //2， 创建数据集并分词
        RDD<String> rdd = spark.sparkContext().textFile("data/ml/crossvalidation/stockNewsTitles_full.txt", 2);
        JavaRDD<Row> parsedRDD = rdd.toJavaRDD().map(line -> {
            String[] splits = line.split("=");
            Integer label = Integer.parseInt(splits[0]);

//            String[] words = WordCountAnalyse.splitWord(splits[1]);
//            for (String word : words) {
//                System.out.print(word + "\t");
//            }
//            System.out.println();
            return RowFactory.create(label, WordCountAnalyse.splitWord(splits[1]));
        });


        // 第二步，动态构造元数据
        // 比如说，id、name等，field的名称和类型，可能都是在程序运行过程中，动态从mysql db里
        // 或者是配置文件中，加载出来的，是不固定的
        // 所以特别适合用这种编程的方式，来构造元数据
        StructType structType = new StructType(new StructField [] {
                new StructField("label", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("message", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });

        Dataset<Row> msgDF = spark.createDataFrame(parsedRDD, structType);
        //3， 数据集分割
        Dataset<Row>[] needFitData = msgDF.randomSplit(new double[]{0.95, 0.05});
        Dataset<Row> trainingData = needFitData[0];
        Dataset<Row> testData = needFitData[1];

//
//        //4， 将标签转化为索引值
//        StringIndexerModel labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(msgDF);
//        //5， 创建Word2Vec，分词向量大小100
//        final int VECTOR_SIZE = 100;
//        Word2Vec word2Vec = new Word2Vec().setInputCol("message").setOutputCol("features").setVectorSize(VECTOR_SIZE).setMinCount(1);
//
//        //6， 创建多层感知器
//        //输入层VECTOR_SIZE个，中间层两层分别是6，,5个神经元，输出层2个
//        int[] layers = new int[] {VECTOR_SIZE, 6, 5, 4};
//        MultilayerPerceptronClassifier mlpc = new MultilayerPerceptronClassifier()
//                .setLayers(layers)
//                //.setBlockSize(512)
//                //.setSeed(1234L)
//                //.setMaxIter(128)
//                .setFeaturesCol("features")
//                .setLabelCol("indexedLabel")
//                .setPredictionCol("prediction");
//
//        //7， 将索引转换为原有标签
//        IndexToString labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels());
//
//        //8， 创建pipeline并训练数据
//        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{labelIndexer, word2Vec, mlpc, labelConverter});
//        PipelineModel model = pipeline.fit(trainingData);
//
//        Dataset<Row> predictionResultDF = model.transform(testData);
//
//        //below 2 lines are for debug use
//        predictionResultDF.printSchema();
//        predictionResultDF.select("message", "label", "predictedLabel").show(300);
//
//        //9， 评估训练结果
//        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
//                .setLabelCol("indexedLabel")
//                .setPredictionCol("prediction")
//                .setMetricName("accuracy");
//
//        double predictionAccuracy = evaluator.evaluate(predictionResultDF);
//        System.out.println("Testing Accuracy is %2.4f".format(String.valueOf(predictionAccuracy * 100)) + "%");
//
//
//        // Input data: Each row is a bag of words from a sentence or document.
//        List<Row> data = Arrays.asList(
//                RowFactory.create(1,WordCountAnalyse.splitWord("年内9家上市公司因财务造假被处罚")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("雷声滚滚！刚刚两家公司确定退市，事涉IPO造假、并购掉坑，18万股东中招")),
//                RowFactory.create(1, WordCountAnalyse.splitWord("安永派驻强大反舞弊团队后 瑞幸造假东窗事发")),
//                RowFactory.create(1, WordCountAnalyse.splitWord("自曝财务造假 瑞幸咖啡暴跌75.57%")),
//                RowFactory.create(1, WordCountAnalyse.splitWord("身陷财务造假漩涡 易见股份大跌后收问询函")),
//
//                RowFactory.create(1,WordCountAnalyse.splitWord("三峡人寿违规销售产品组合 被叫停新产品备案六个月")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("又一股跌停！ST群兴因信披违规遭立案调查")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("银河证券：海南自由贸易港迎长期投资机会，短期利好旅游业复苏")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("小微金融政策出炉，中小银行迎结构性利好")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("鹏博士大涨5.16% 首季净利润增长52.77%-60.52%")),
//                RowFactory.create(1,WordCountAnalyse.splitWord("智飞生物大涨5.30% 首季净利润增长0-10.00%"))
//        );
//        StructType schema = new StructType(new StructField[]{
//                new StructField("label", DataTypes.IntegerType, false, Metadata.empty()),
//                new StructField("message", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
//        });
//        Dataset<Row> testTitleDS = spark.createDataFrame(data, schema);
//        // 需要进一步将切分单词的数据，转换为特征向量
//
//        // TODO 2.使用训练好的模型测试样例数据。使用持久化后的PipelineModel对原始数据进行预测
//        Dataset<Row> res = model.transform(testTitleDS);
//        res.printSchema();
//        res.select("message", "label", "predictedLabel").show(false);


        //模型训练器（模型学习器，算法）
        LogisticRegression lr = new LogisticRegression();
        //这是一个管道，管道其实就是一个模型学习器，这个管道有3个stage
        //Pipeline newPipeline = new Pipeline()
        //        .setStages(new PipelineStage[]{tokenizer, hashingTF, lr});


        //4， 将标签转化为索引值
        StringIndexerModel labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(msgDF);
        //5， 创建Word2Vec，分词向量大小100
        final int VECTOR_SIZE = 100;
        Word2Vec word2Vec = new Word2Vec().setInputCol("message").setOutputCol("features").setVectorSize(VECTOR_SIZE).setMinCount(1);

        //6， 创建多层感知器
        //输入层VECTOR_SIZE个，中间层两层分别是6，,5个神经元，输出层2个
        int[] layers = new int[] {VECTOR_SIZE, 6, 5, 4};
        MultilayerPerceptronClassifier mlpc = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                //.setBlockSize(512)
                //.setSeed(1234L)
                //.setMaxIter(128)
                .setFeaturesCol("features")
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction");

        //7， 将索引转换为原有标签
        IndexToString labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels());

        Pipeline newPipeline = new Pipeline().setStages(new PipelineStage[]{labelIndexer, word2Vec, mlpc, labelConverter});
        //PipelineModel model = pipeline.fit(trainingData);


        //todo 为了得到最佳模型，我们需要进行交叉验证，参数的交叉验证，和数据集的交叉验证（）
        mlpc.setLayers(new int[] {VECTOR_SIZE, 6, 5, 4});

        ParamMap[] paramMaps = new ParamGridBuilder()
                .addGrid(mlpc.blockSize(), new int[]{1024})
                .addGrid(mlpc.seed(), new long[]{100, 200})
                .addGrid(lr.maxIter(), new int[]{40,100})
                .addGrid(lr.regParam(), new double[]{0.01})
                .build();

        //创建模型评估器

        MulticlassClassificationEvaluator multiclassClassificationEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        //模型学习器
        CrossValidator crossValidator = new CrossValidator()
                //设置模型学习器，交叉验证的对象
                .setEstimator(newPipeline)
                //参数交叉验证
                .setEstimatorParamMaps(paramMaps)
                //数据集交叉验证，数据集划分几个等分
                .setNumFolds(4)
                //设置模型评估器
                .setEvaluator(multiclassClassificationEvaluator);

        //交叉验证模型，可以其进行预测我们的样本数据
        //模型学习器调用fit方法后会产生一个转换器，这个crossValidatorModel就是一个转换器
        CrossValidatorModel crossValidatorModel = crossValidator.fit(trainingData);
        //todo 获取最佳模型
        Model<?> bestModel = crossValidatorModel.bestModel();
        //获取管道中hashingTF转换器最佳参数
        System.out.println("LogisticRegression模型学习器参数 " + bestModel.params());

        //获取管道中逻辑回归LogisticRegression模型学习器最佳参数
        System.out.println("LogisticRegression模型学习器最佳参数 " + bestModel.explainParams());

        Dataset<Row> predictedDF = bestModel.transform(testData);

        //below 2 lines are for debug use
        predictedDF.printSchema();
        predictedDF.select("message", "label", "predictedLabel").show(300);

        //9， 评估训练结果
        MulticlassClassificationEvaluator newEvaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = newEvaluator.evaluate(predictedDF);
        System.out.println("Testing accuracy is %2.4f".format(String.valueOf(accuracy * 100)) + "%");
    }
}


