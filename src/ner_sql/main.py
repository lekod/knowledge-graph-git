from rec_entity_sql import EntityRec
from entity_cleaner_sql import Abbreviation
from entity_cleaner_sql import Similarity

if __name__ == ("__main__"):
    del_abbr = Abbreviation()
    entity_rec = EntityRec()
    check_similar = Similarity()

    # Delete all abbrevations to avoid double nodes
    del_abbr.multithread_abbr()

    # perform NER
    entity_rec.store_processed_urls_to_file()
    entity_rec.multithread_ner()
    entity_rec.org_counter("org_counter", "org_counter_list.csv")
    entity_rec.pers_counter("pers_counter", "pers_counter_list.csv")

    #Clean out similar orgs
    check_similar.multithread_sim()
