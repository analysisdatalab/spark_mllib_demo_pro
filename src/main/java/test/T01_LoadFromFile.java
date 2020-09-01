package test;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * @Author:
 * @Date: 2020-07-24 14:29
 * @Version: 1.0
 * @Modified By:
 * @Description:
 */
public class T01_LoadFromFile {
    public static void main(String[] args) {
        loadFromFile("data\\ml\\crossvalidation\\SMSSpamCollection", null);
    }

    // try-with-resources可以优雅关闭文件，异常时自动关闭文件
    public static String loadFromFile(String path, String charset) {
        String resText = "";
        if (charset == null) {
            charset = "utf-8";
        }
        InputStreamReader reader = null;
        int lineNo = 0;
        try {
            reader = new InputStreamReader(new FileInputStream(path), charset);
            BufferedReader br = new BufferedReader(reader);
            String line;
            while ((line = br.readLine()) != null) {
                String[] split = line.split(" ");
                lineNo++;
                if (split[0].equals("ham") || split[0].equals("spam")) {

                } else {
                    System.out.println(line + " lineNo:" + lineNo);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        return resText;
    }
}
