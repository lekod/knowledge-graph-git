# The models are used to structure the Nodes and relationships in a certain way from queries_sql.py

from neomodel import (
    StructuredNode,
    StringProperty,
    RelationshipTo,
    DateProperty,
    StructuredRel,
    IntegerProperty,
)

# Creating relationships

class Interaction(StructuredRel):
    weight = IntegerProperty()
    name = StringProperty()


# Creating nodes

class stakeholder_pers(StructuredNode):
    name = StringProperty(required=True, unique_index=True)
    link = StringProperty(required=True, unique_index=True)
    weight = IntegerProperty()

class stakeholder_org(StructuredNode):
    name = StringProperty(required=True, unique_index=True)
    link = StringProperty(required=True, unique_index=True)
    weight = IntegerProperty()

class second_website(StructuredNode):
    name = StringProperty(required=True)
    link = StringProperty(required=True, unique_index=True)
    links_to_sp = RelationshipTo(stakeholder_pers, "LINKS")
    links_to_so = RelationshipTo(stakeholder_org, "LINKS")
    date_up = DateProperty()

class first_website(StructuredNode):
    name = StringProperty(required=True)
    link = StringProperty(required=True, unique_index= True)
    links_to = RelationshipTo(second_website, "LINKS")




