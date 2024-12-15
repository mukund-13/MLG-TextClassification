import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# Add documents
documents = ["doc_1", "doc_2", "doc_3", "doc_4"]
for d in documents:
    G.add_node(d, color='blue')  # all documents in blue
# Add authors
authors = ["author_1", "author_2", "author_3"]
for a in authors:
    G.add_node(a, color='red')   # authors in red
# Add tags
tags = ["tag_news", "tag_sports", "tag_business", "tag_tech"]
for t in tags:
    G.add_node(t, color='green') # tags in green

# Create some edges:
# doc_1 is connected to author_1 and tag_news, tag_tech
G.add_edge("doc_1", "author_1")
G.add_edge("doc_1", "tag_news")
G.add_edge("doc_1", "tag_tech")

# doc_2 connected to author_1 (same author as doc_1), and tag_sports
G.add_edge("doc_2", "author_1")
G.add_edge("doc_2", "tag_sports")

# doc_3 connected to author_2 and tag_business
G.add_edge("doc_3", "author_2")
G.add_edge("doc_3", "tag_business")

# doc_4 connected to author_3, and shares tag_tech with doc_1 and tag_business with doc_3
G.add_edge("doc_4", "author_3")
G.add_edge("doc_4", "tag_tech")
G.add_edge("doc_4", "tag_business")

# optional: Add some extra author-to-document or tag-to-document edges to show complexity:
G.add_edge("doc_2", "tag_tech")       # doc_2 also associated with tech
G.add_edge("doc_3", "tag_sports")     # doc_3 also has sports interest
G.add_edge("doc_4", "tag_news")       # doc_4 also connected to news

pos = nx.spring_layout(G, seed=42)

# Extract color attributes for nodes
node_colors = [G.nodes[n]['color'] for n in G.nodes]

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos)
plt.title("Heterogeneous Graph (Documents, Authors, Tags)")
plt.axis('off')
plt.savefig("graph2.png")
