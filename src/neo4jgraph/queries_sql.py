# These queries have to be executed to construct a Network from the SQL database in Neo4J

from neomodel import db
def create_base_websites(cursor):
# Build all base websites that are the ministries from scratch

    query = """
        WITH [
        {link: 'https://www.stmd.bayern.de', name: 'Bayerisches Staatsministerium für Digitales'},
        {link: 'https://www.stmi.bayern.de', name: 'Bayerisches Staatsministerium des Inneres, für Sport und Integration'},
        {link: 'https://www.stmwk.bayern.de', name: 'Bayerisches Staatsministerium für Wissenschaft und Kunst'},
        {link: 'https://www.stmb.bayern.de', name: 'Bayerisches Staatsministerium für Wohnen, Bau und Verkehr'}, 
        {link: 'https://www.justiz.bayern.de', name: 'Bayerisches Staatsministerium der Justiz'},
        {link: 'https://www.km.bayern.de', name: 'Bayerisches Staatsministerium für Unterricht und Kultus'},
        {link: 'https://www.stmfh.bayern.de', name: 'Bayerisches Staatsministerium für Finanzen und Heimat'}, 
        {link: 'https://www.stmwi.bayern.de', name: 'Bayerisches Staatsministerium für Wirtschaft, Landesentwicklung und Energie'},
        {link: 'https://www.stmuv.bayern.de', name: 'Bayerisches Staatsministerium für Umwelt und Verbraucherschutz'},
        {link: 'https://www.stmelf.bayern.de', name: 'Bayerisches Staatsministerium für Ernährung, Landwirtschaft, Forsten und Tourismus'},
        {link: 'https://www.stmas.bayern.de', name: 'Bayerisches Staatsministerium für Familie, Arbeit und Soziales'},
        {link: 'https://www.stmgp.bayern.de', name: 'Bayerisches Staatsministerium für Gesundheit, Pflege und Prävention'}
        ] AS baseList
        
        FOREACH (entry IN baseList |
        MERGE (bw:base_websites {link: entry.link})
        SET bw.name = entry.name
        );
    """

    results, meta = db.cypher_query(query)

# Import all second_websites that connect the stakeholders with the Ministries
def create_second_websites(cursor):

    cursor.execute("SELECT second_website, second_title, date FROM scraped_data")
    for row in cursor.fetchall():
        second_website = row[0]  # Access the first column (first_website)
        second_title = row[1][:50]   # Access the second column (first_title)
        date = row[2]

        # Construct the Cypher query using the fetched data
        query = """
            MERGE (sw:second_website {link: $second_website})
            SET sw.name = $second_title,
                sw.date_up = $date
        """
        parameters = {"second_website": second_website, "second_title": second_title, "date": date}

        results, meta = db.cypher_query(query, parameters)

# Build the relationship from base website to secondwebsite that are in one line
def rel_basewebsite_to_secondwebsite(cursor):

    cursor.execute("SELECT first_website, second_website FROM scraped_data")
    for row in cursor.fetchall():
        first_website = row[0]
        second_website = row[1]
        query = """
        MATCH (from:base_websites {link: $first_website})
        MATCH (to:second_website {link: $second_website})
        MERGE (from)-[:LINKS_TO]->(to);
        """
        parameters = {"first_website": first_website, "second_website": second_website}

        results, meta = db.cypher_query(query, parameters)

# Construct the relationship between the websites that are not base websites
def rel_firstwebsite_to_secondwebsite(cursor):

    cursor.execute("SELECT first_website, second_website FROM scraped_data")
    for row in cursor.fetchall():
        first_website = row[0]  # Access the first column (first_website)
        second_website = row[1]   # Access the second column (first_title)
        query = """
        MATCH (from:second_website {link: $first_website})
        MATCH (to:second_website {link: $second_website})
        MERGE (from)-[:LINKS_TO]->(to);
        """
        parameters = {"first_website": first_website, "second_website": second_website}

        results, meta = db.cypher_query(query, parameters)

# Create the stakeholder Persons and give them a date of the scraped website
def create_stakeholder_names(cursor_stake):
    cursor_stake.execute("SELECT persons, date FROM ner_data_sim")
    for row in cursor_stake.fetchall():
        persons = row[0]
        date = row[1]
        pers_names = [entry.strip("'{}'") for entry in persons.split(",")]

        for person in pers_names:

            pers_name_clean = person.replace("'", "")
            query = """
            WITH $personName AS persName, trim($date) AS trimmedDateUp
            WHERE trimmedDateUp <> ''
            WITH persName, collect(trimmedDateUp) AS nonEmptyDates
            WITH persName, apoc.coll.min([date IN nonEmptyDates WHERE date IS NOT NULL | date]) AS minDateUp
            MERGE (sp:stakeholder_pers {name: persName})
            SET sp.date_up = minDateUp
            RETURN sp;
            """
            parameters = {"personName": pers_name_clean, "date": date}
            results, meta = db.cypher_query(query, parameters)

# Create the stakeholder Organisations and give them a date of the website they are referred on

def create_stakeholder_orgs(cursor_stake):
    cursor_stake.execute("SELECT organisations, date FROM ner_data_sim")
    for row in cursor_stake.fetchall():
        organisations = row[0]
        date = row[1]
        orgs_names = [entry.strip("'{}'") for entry in organisations.split(",")]

        for org in orgs_names:

            org_name_clean = org.replace("'", "")
            cursor_stake.execute("SELECT COUNT(*) FROM ner_data_sim WHERE organisations LIKE ?", ('%' + org + '%',))
            count = cursor_stake.fetchone()[0]

            # If the organization name occurs at least 5 times, create the node
            if count >=2:
                query = """
                WITH $orgName AS orgName, trim($date) AS trimmedDateUp
                WHERE trimmedDateUp <> ''
                WITH orgName, collect(trimmedDateUp) AS nonEmptyDates
                WITH orgName, apoc.coll.min([date IN nonEmptyDates WHERE date IS NOT NULL | date]) AS minDateUp
                MERGE (so:stakeholder_org {name: orgName})
                SET so.date_up = minDateUp, so.count = $count
                RETURN so;
                """
                parameters = {"orgName": org_name_clean, "date": date, "count": count}
                results, meta = db.cypher_query(query, parameters)

# Construct a relationship from the websites to the stakeholder persons
def rel_secondwebsite_to_person(cursor_stake):
    cursor_stake.execute("SELECT second_website, persons FROM ner_data_sim")
    for row in cursor_stake.fetchall():
        second_website = row[0]
        persons = row[1]
        pers_names = [entry.strip("'{}'") for entry in persons.split(",")]

        for pers in pers_names:
            if pers != 'set()':
                pers_name_clean = pers.replace("'", "")
                query = """
    
                    MATCH (from: second_website {link: $second_website})
                    MATCH (to: stakeholder_pers {name: $personName})
                    MERGE (from)-[:LINKS_TO]->(to);
                    """
                parameters = {"second_website": second_website, "personName": pers_name_clean}
                results, meta = db.cypher_query(query, parameters)

# Construct a relationship from the websites to the stakeholder organisations
def rel_secondwebsite_to_org(cursor_stake):
    cursor_stake.execute("SELECT second_website, organisations FROM ner_data_sim")
    for row in cursor_stake.fetchall():
        second_website = row[0]
        organisations = row[1]
        org_names = [entry.strip("'{}'") for entry in organisations.split(",")]

        for org in org_names:
            if org != 'set()':
                query = """
                        MATCH (from: second_website {link: $second_website})
                        MATCH (to: stakeholder_org {name: $orgName})
                        MERGE (from)-[:LINKS_TO]->(to);
                        """
                parameters = {"second_website": second_website, "orgName": org}
                results, meta = db.cypher_query(query, parameters)

# Construct a relationship from the stakeholder organisations to the stakeholder persons
def rel_person_to_org(cursor_stake):
    cursor_stake.execute("SELECT persons, organisations FROM ner_data_sim")
    for row in cursor_stake.fetchall():
        persons = row[0]
        organisations = row[1]
        pers_names = [entry.strip("'{}'") for entry in persons.split(",")]
        org_names = [entry.strip("'{}'") for entry in organisations.split(",")]

        for org in org_names:
            for pers in pers_names:
                if org and pers != 'set()':
                    pers_name_clean = pers.replace("'", "")
                    query = """
                                MATCH (from:stakeholder_org {name: $orgName})
                                MATCH (to:stakeholder_pers {name: $persName})
                                MERGE (from)-[:LINKS_TO]->(to);
                                """
                    parameters = {"persName": pers, "orgName": org}
                    results, meta = db.cypher_query(query, parameters)


# These queries can be executed directly in Neo4J Browser to visualise the graph in a certain way
# Show every link of the network
def show_whole_links():
    query = """
    MATCH (source: base_websites)-[:LINKS_TO]->(target:second_website)
    RETURN source, target
    UNION
    MATCH (source:first_website)-[:LINKS_TO]->(target:second_website)
    RETURN source, target
    UNION
    MATCH (source:second_website)-[:LINKS_TO]->(target:stakeholder_pers)
    RETURN source, target
    UNION
    MATCH (source:second_website)-[:LINKS_TO]->(target:stakeholder_org)
    RETURN source, target
    UNION
    MATCH (source:stakeholder_org)-[:LINKS_TO]->(target:stakeholder_pers)
    RETURN source, target;
    
    """

# Visualize the StMD network table
def show_stmd_pers_table():
    query = """
    MATCH (s:second_website)-[r:LINKS_TO]->(t:stakeholder_pers)
    WHERE s.link STARTS WITH "https://www.stmd."
    WITH s, t, count(*) AS Count
    RETURN s.link, t.name AS stakeholder, Count
    """

# Visualize the StMD Graph
def show_stmd_graph():
    query = """
    MATCH p=(base:base_websites)-[:LINKS_TO]->(s:second_website)-[r:LINKS_TO]->(t)
    WHERE (s.link STARTS WITH "https://www.stmd." AND base.name = "Bayerisches Staatsministerium für Digitales") 
    AND (t:stakeholder_org OR t:stakeholder_pers)
    RETURN p
    """

# Delete weird nodes
def delete_errors():
    query = """
    MATCH (s:stakeholder_org)
    WHERE s.name IS NULL OR s.name = "" OR s.name = "&"
    DETACH DELETE s
    """
    results, meta = db.cypher_query(query)

# Show the whole network
def show_all():
    query = """
    MATCH (n)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN n, r, m;
    """

