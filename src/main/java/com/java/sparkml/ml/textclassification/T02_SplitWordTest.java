package com.java.sparkml.ml.textclassification;

import org.wltea.analyzer.app.WordCountAnalyse;

/**
 * @Author:
 * @Date: 2020-07-24 17:02
 * @Version: 1.0
 * @Modified By:
 * @Description:
 */
public class T02_SplitWordTest {
    public static void main(String[] args) throws Exception {
        String str = "中国人民共和国国歌";
        String[] words = WordCountAnalyse.splitWord(str);
        for (String word : words) {
            System.out.println(word);
        }
        System.out.println();
    }
}
