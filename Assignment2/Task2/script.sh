hadoop jar /home/cse587/hadoop-3.1.2/share/hadoop/tools/lib/hadoop-streaming-3.1.2.jar -file /home/cse587/mapperTask2.py -mapper /home/cse587/mapperTask2.py -file /home/cse587/reducerTask2.py -reducer /home/cse587/reducerTask2.py -input /home/pa2/gutenberg -output /home/pa2/videoOutputs/taskTwo1 -jobconf mapred.reduce.tasks=5
hadoop jar /home/cse587/hadoop-3.1.2/share/hadoop/tools/lib/hadoop-streaming-3.1.2.jar -file /home/cse587/mapperTask2_1.py -mapper /home/cse587/mapperTask2_1.py -file /home/cse587/reducerTask2.py -reducer /home/cse587/reducerTask2.py -input /home/pa2/videoOutputs/taskTwo1/ -output /home/pa2/videoOutputs/taskTwo2
hadoop fs -copyToLocal /home/pa2/videoOutputs/taskTwo1/ /home/cse587/videoOutputs/
hadoop fs -copyToLocal /home/pa2/videoOutputs/taskTwo2/ /home/cse587/videoOutputs/


