from elasticsearch import Elasticsearch

#es =  Elasticsearch(['https://elastic:VYkTJ4EG6rv2xejidHMf@localhost:9200'])
es = Elasticsearch(
    ['localhost'],
    http_auth=('elastic', 'VYkTJ4EG6rv2xejidHMf'),
    port=9200,
)
x = es.search(index="bitcoin-prices", body={"query": {"fuzzy_like_this_field" : { "High": {"like_text": "9"}}}})['hits']

print(x)