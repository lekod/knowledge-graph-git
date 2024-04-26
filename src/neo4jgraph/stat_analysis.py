
# We need the plugin Graph Data Science (GDS) library to run in docker
# Define graph name
# These queries have to be run directly in Neo4J Browser
def name_graph():
    # this works
    query = """
    CALL gds.graph.project(
  'KnowledgeG', 
  {
    base_websites: {
      label: 'base_websites'
    },
    second_website: {
      label: 'second_website'
    },
    stakeholder_org: {
      label: 'stakeholder_org'
    },
    stakeholder_pers: {
      label: 'stakeholder_pers'
    }
  }, 
  {
    LINKS_TO: {
      type: 'LINKS_TO'
    }
  }
);
    """

# Define the stakeholder graph
def name_stakeholder_graph():
    query = """
    CALL gds.graph.project(
  'ClusterAnalysis', 
  {
    stakeholder_org: {
      label: 'stakeholder_org'
    },
    stakeholder_pers: {
      label: 'stakeholder_pers'
    }
  }, 
  {
    LINKS_TO: {
      type: 'LINKS_TO'
    }
  }
);
    """

# Make a cluster Analysis of the network
def cluster_analysis():

    query = """
    CALL gds.louvain.stream('ClusterAnalysis', { maxIterations: 10, minCommunitySize: 10, tolerance: 0.1 })
    YIELD nodeId, communityId
    WITH communityId, COLLECT(nodeId) AS communityMembers
    WITH communityId, communityMembers[0..5] AS representativeIds
    WITH communityId, COLLECT([id IN representativeIds | gds.util.asNode(id).name]) AS representatives
    RETURN communityId, representatives
    """

# Count the Indegree of the network nodes
def indegree():
    query = """
    CALL gds.degree.stream('KnowledgeG', { orientation: 'REVERSE' })
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).id AS id, gds.util.asNode(nodeId).name AS name, score
    ORDER BY score DESC;
    """

    # Show in graph the stakeholder with an indegree of at least 3:
    query = """
    CALL gds.degree.stream('KnowledgeG', { orientation: 'REVERSE' })
    YIELD nodeId, score
    WITH gds.util.asNode(nodeId) AS node, score
    WHERE score > 3
    MATCH (node)-[rel]->(relatedNode)
    RETURN node, rel, relatedNode;
    """



def betweenness_centrality():

    # Return the stakeholder_org/ stakeholder_pers with the highest betweenness centrality
    query = """
    CALL {
    CALL gds.betweenness.stream('KnowledgeG', {})
    YIELD nodeId, score
    WITH gds.util.asNode(nodeId) AS node, score
    WHERE 'stakeholder_org' IN labels(node)
    RETURN node.name AS name, score
    ORDER BY score DESC
    LIMIT 20
    }
    RETURN name, score;
    """

    # show in graph over 4:
    query = """
    CALL gds.betweenness.stream('KnowledgeG', {})
    YIELD nodeId, score
    WITH gds.util.asNode(nodeId) AS node, score
    WHERE 'stakeholder_org' IN labels(node) AND score > 4
    MATCH (node)-[rel]->(relatedNode)
    RETURN node, rel, relatedNode;
    """

def eigenvector_centrality():
    # Calculate the eigenvector centrality
    query = """
    CALL gds.eigenvector.write('KnowledgeG', { writeProperty: 'eigenvectorCentrality' })
    YIELD nodePropertiesWritten

    MATCH (n)
    RETURN n.name AS nodeName, n.eigenvectorCentrality AS eigenvectorCentrality
    ORDER BY eigenvectorCentrality DESC
    LIMIT 20
    """

    query = """
    MATCH (n)
    WHERE n.eigenvectorCentrality > 0.01
    RETURN n;
    """

# Calculate the distribution of the degree within the network.
def degreeDistribution():
    query = """
    CALL gds.graph.list('KnowledgeG')
    YIELD graphName, degreeDistribution;
    """


