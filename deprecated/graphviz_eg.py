import graphviz as gv

# Basic Digraph
dag = gv.Digraph(comment="Basic Graph")
dag.node('1', '1')
dag.node('2', '2')
dag.node('3', '3')
dag.node('4', '4')

dag.edges(['12', '13', '24', '34'])
dag.render('test', view=True)

