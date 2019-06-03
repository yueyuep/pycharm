import tensorflow as tf
x=tf.Variable(3,name="x")
y=tf.Variable(4,name="y")
f=x*y
session=tf.Session()
"""
vay1:


session.run(x.initializer)
session.run(y.initializer)
print(session.run(f))
"""


"""
vay2

with session as sess:
    x.initializer.run()
    y.initializer.run()
    print(f.eval())
"""

"""
vay3
variable=tf.global_variables_initializer()
with session as sess:
    #初始化所有的图谱中的变量
    variable.run()
    print(y.eval())

"""

# variables=tf.global_variables_initializer()
# sess=tf.InteractiveSession()
# variables.run()
# print(f.eval())
# sess.close()

graph=tf.Graph()
with graph.as_default():
    z=tf.Variable(6,name="z")
print(z.graph is graph)
print(z.graph is tf.get_default_graph())
