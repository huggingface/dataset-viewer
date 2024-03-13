# Filter rows in a dataset

Datasets Server provides a `/filter` endpoint for filtering rows in a dataset.

<Tip warning={true}>
  Currently, only <a href="./parquet">datasets with Parquet exports</a>
  are supported so Datasets Server can index the contents and run the filter query without
  downloading the whole dataset.
</Tip>

This guide shows you how to use Datasets Server's `/filter` endpoint to filter rows based on a query string.
Feel free to also try it out with [ReDoc](https://redocly.github.io/redoc/?url=https://datasets-server.huggingface.co/openapi.json#operation/filterRows).

The `/filter` endpoint accepts the following query parameters:
- `dataset`: the dataset name, for example `nyu-mll/glue` or `mozilla-foundation/common_voice_10_0`
- `config`: the configuration name, for example `cola`
- `split`: the split name, for example `train`
- `where`: the filter condition
- `offset`: the offset of the slice, for example `150`
- `length`: the length of the slice, for example `10` (maximum: `100`)

The `where` parameter must be expressed as a comparison predicate, which can be:
- a simple predicate composed of a column name, a comparison operator, and a value
  - the comparison operators are: `=`, `<>`, `>`, `>=`, `<`, `<=`
- a composite predicate composed of two or more simple predicates (optionally grouped with parentheses to indicate the order of evaluation), combined with logical operators
  - the logical operators are: `AND`, `OR`, `NOT`

For example, the following `where` parameter value
```
where=age>30 AND (name='Simone' OR children=0)
```
will filter the data to select only those rows where the float "age" column is larger than 30 and,
either the string "name" column is equal to 'Simone' or the integer "children" column is equal to 0.

<Tip>
  Note that, following SQL syntax, string values in comparison predicates must be enclosed in single quotes,
  for example: <code>'Scarlett'</code>.
  Additionally, if the string value contains a single quote, it must be escaped with another single quote,
  for example: <code>'O''Hara'</code>.
</Tip>

For example, let's filter those rows with no_answer=false in the `train` split of the `SelfRC` configuration of the `ibm/duorc` dataset restricting the results to the slice 150-151:

<inferencesnippet>
<python>
```python
import requests
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://datasets-server.huggingface.co/filter?dataset=ibm/duorc&config=SelfRC&split=train&where=no_answer=true&offset=150&length=2"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
data = query()
```
</python>
<js>
```js
import fetch from "node-fetch";
async function query(data) {
    const response = await fetch(
        "https://datasets-server.huggingface.co/filter?dataset=ibm/duorc&config=SelfRC&split=train&where=no_answer=true&offset=150&length=2",
        {
            headers: { Authorization: `Bearer ${API_TOKEN}` },
            method: "GET"
        }
    );
    const result = await response.json();
    return result;
}
query().then((response) => {
    console.log(JSON.stringify(response));
});
```
</js>
<curl>
```curl
curl https://datasets-server.huggingface.co/filter?dataset=ibm/duorc&config=SelfRC&split=train&where=no_answer=true&offset=150&length=2 \
        -X GET \
        -H "Authorization: Bearer ${API_TOKEN}"
```
</curl>
</inferencesnippet>

The endpoint response is a JSON containing two keys (same format as [`/rows`](./rows)):

- The [`features`](https://huggingface.co/docs/datasets/about_dataset_features) of a dataset, including the column's name and data type.
- The slice of `rows` of a dataset and the content contained in each column of a specific row.

The rows are ordered by the row index.

For example, here are the `features` and the slice 150-151 of matching `rows` of the `ibm/duorc`/`SelfRC` train split for the `where` condition `no_answer=true`:

```json
{
   "features":[
      {
         "feature_idx":0,
         "name":"plot_id",
         "type":{
            "dtype":"string",
            "_type":"Value"
         }
      },
      {
         "feature_idx":1,
         "name":"plot",
         "type":{
            "dtype":"string",
            "_type":"Value"
         }
      },
      {
         "feature_idx":2,
         "name":"title",
         "type":{
            "dtype":"string",
            "_type":"Value"
         }
      },
      {
         "feature_idx":3,
         "name":"question_id",
         "type":{
            "dtype":"string",
            "_type":"Value"
         }
      },
      {
         "feature_idx":4,
         "name":"question",
         "type":{
            "dtype":"string",
            "_type":"Value"
         }
      },
      {
         "feature_idx":5,
         "name":"answers",
         "type":{
            "feature":{
               "dtype":"string",
               "_type":"Value"
            },
            "_type":"Sequence"
         }
      },
      {
         "feature_idx":6,
         "name":"no_answer",
         "type":{
            "dtype":"bool",
            "_type":"Value"
         }
      }
   ],
   "rows":[
      {
         "row_idx":12825,
         "row":{
            "plot_id":"/m/06qxsf",
            "plot":"Prologue\nA creepy-looking coroner introduces three different horror tales involving his current work on cadavers in \"body bags\".\n\"The Gas Station\"[edit]\nAnne is a young college student who arrives for her first job working the night shift at an all-night filling station near Haddonfield, Illinois (a reference to the setting of Carpenter's two Halloween films). The attending worker, Bill, tells her that a serial killer has broken out of a mental hospital, and cautions her not to leave the booth at the station without the keys because the door locks automatically. After Bill leaves, Anne is alone and the tension mounts as she deals with various late-night customers seeking to buy gas for a quick fill-up, purchase cigarettes or just use the restroom key, unsure whether any of them might be the escaped maniac. Eventually, when Anne suspects that the escaped killer is lurking around the gas station, she tries to call the police, only to find that the phone line is dead. Soon after that, she finds an elaborately grotesque drawing in the Restroom and then the dead body of a transient sitting in a pickup truck on the lift in one of the garage bays. She makes a phone call for help which results in her realization that \"Bill\", the attending worker she met earlier, is in fact the escaped killer, who has killed the real Bill and is killing numerous passers-by. She finds the real Bill's dead body in one of the lockers. Serial Killer \"Bill\" then reappears and attempts to kill Anne with a machete, breaking into the locked booth by smashing out the glass with a sledgehammer and then chasing her around the deserted garage. Just as he is about to kill her, a customer returns, having forgotten his credit card, and he wrestles the killer, giving Anne time to crush him under the vehicle lift.\n\"Hair\"[edit]\nRichard Coberts is a middle-aged businessman who is very self-conscious about his thinning hair. This obsession has caused a rift between him and his long-suffering girlfriend Megan. Richard answers a television ad about a \"miracle\" hair transplant operation, pays a visit to the office, and meets the shady Dr. Lock, who, for a very large fee, agrees to give Richard a surgical procedure to make his hair grow back. The next day, Richard wakes up and removes the bandage around his head, and is overjoyed to find that he has a full head of hair. But soon he becomes increasingly sick and fatigued, and finds his hair continuing to grow and, additionally, growing out of parts of his body, where hair does not normally grow. Trying to cut some of the hair off, he finds that it \"bleeds\", and, examining some of the hairs under a magnifying glass, sees that they are alive and resemble tiny serpents. He goes back to Dr. Lock for an explanation, but finds himself a prisoner as Dr. Lock explains that he and his entire staff are aliens from another planet, seeking out narcissistic human beings and planting seeds of \"hair\" to take over their bodies for consumption as part of their plan to spread their essence to Earth.\n\"Eye\"[edit]\nBrent Matthews is a baseball player whose life and career take a turn for the worse when he gets into a serious car accident in which his right eye is gouged out. Unwilling to admit that his career is over, he jumps at the chance to undergo an experimental surgical procedure to replace his eye with one from a recently deceased person. But soon after the surgery he begins to see things out of his new eye that others cannot see, and begins having nightmares of killing women and having sex with them. Brent seeks out the doctor who operated on him, and the doctor tells him that the donor of his new eye was a recently executed serial killer and necrophile who killed several young women, and then had sex with their dead bodies. Brent becomes convinced that the spirit of the dead killer is taking over his body so that he can resume killing women. He flees back to his house and tells his skeptical wife, Cathy, about what is happening. Just then the spirit of the killer emerges and attempts to kill Cathy as well. Cathy fights back, subduing him long enough for Brent to re-emerge. Realizing that it is only a matter of time before the killer emerges again, Brent cuts out his donated eye, severing his link with the killer, but then bleeds to death.\nEpilogue The coroner is finishing telling his last tale when he hears a noise from outside the morgue. He crawls back inside a body bag, revealing that he himself is a living cadaver, as two other morgue workers begin to go to work on his \"John Doe\" corpse.",
            "title":"John Carpenter presents Body Bags",
            "question_id":"cf58489f-12ba-ace6-67a7-010d957b4ff4",
            "question":"What happens soon after the surgery?",
            "answers":[
               
            ],
            "no_answer":true
         },
         "truncated_cells":[
            
         ]
      },
      {
         "row_idx":12836,
         "row":{
            "plot_id":"/m/04z_3pm",
            "plot":"In 1976, eight-year-old Mary Daisy Dinkle (Bethany Whitmore) lives a lonely life in Mount Waverley, Australia. At school, she is teased by her classmates because of an unfortunate birthmark on her forehead; while at home, her distant father, Noel, and alcoholic, kleptomaniac mother, Vera, provide little support. Her only comforts are her pet rooster, Ethel; her favourite food, sweetened condensed milk; and a Smurfs-like cartoon show called The Noblets. One day, while at the post office with her mother, Mary spots a New York City telephone book and, becoming curious about Americans, decides to write to one. She randomly chooses Max Jerry Horowitz's name from the phone book and writes him a letter telling him about herself, sending it off in the hope that he will become her pen friend.\nMax Jerry Horowitz (Philip Seymour Hoffman) is a morbidly obese 44-year-old ex-Jewish atheist who has trouble forming close bonds with other people, due to various mental and social problems. Though Mary's letter initially gives him an anxiety attack, he decides to write back to her, and the two quickly become friends (partly due to their shared love of chocolate and The Noblets). Due to Vera's disapproval of Max, Mary tells him to send his letters to her agoraphobic neighbour, Len Hislop, whose mail she collects regularly. When Mary later asks Max about love, he suffers a severe anxiety attack and is institutionalized for eight months. After his release, he is hesitant to write to Mary again for some time. On his 48th birthday, he wins the New York lottery, using his winnings to buy a lifetime supply of chocolate and an entire collection of Noblet figurines. He gives the rest of his money to his elderly neighbour Ivy, who uses most of it to pamper herself before dying in an accident with a malfunctioning jet pack. Meanwhile, Mary becomes despondent, thinking Max has abandoned her.\nOn the advice of his therapist, Max finally writes back to Mary and explains he has been diagnosed with Asperger syndrome. Mary is thrilled to hear from him again, and the two continue their correspondence for the next several years. When Noel retires from his job at a tea bag factory, he takes up metal detecting, but is soon swept away (and presumably killed) by a big tidal bore while on a beach. Mary (Toni Colette) goes to university and has her birthmark surgically removed, and develops a crush on her Greek Australian neighbour, Damien Popodopoulos (Eric Bana). Drunk and guilt-ridden over her husband's death, Vera accidentally kills herself after she drinks embalming fluid (which she mistook for cooking sherry). Mary and Damien grow closer following Vera's death and are later married.\nInspired by her friendship with Max, Mary studies psychology at university, writing her doctoral dissertation on Asperger syndrome with Max as her test subject. She plans to have her dissertation published as a book; but when Max receives a copy from her, he is infuriated that she has taken advantage of his condition, which he sees as an integral part of his personality and not a disability that needs to be cured. He breaks off communication with Mary (by removing the letter \"M\" from his typewriter), who, heartbroken, has the entire run of her book pulped, effectively ending her budding career. She sinks into depression and begins drinking cooking sherry, as her mother had done. While searching through a cabinet, she finds a can of condensed milk, and sends it to Max as an apology. She checks the post daily for a response and one day finds a note from Damien, informing her that he has left her for his own pen friend, Desmond, a sheep farmer in New Zealand.\nMeanwhile, after an incident in which he nearly chokes a homeless man (Ian \"Molly\" Meldrum) in anger, after throwing a used cigarette, Max realizes Mary is an imperfect human being, like himself, and sends her a package containing his Noblet figurine collection as a sign of forgiveness. Mary, however, has sunken into despair after Damien's departure, and fails to find the package on her doorstep for several days. Finding some Valium that had belonged to her mother, and unaware that she is pregnant with Damien's child, Mary decides to commit suicide. As she takes the Valium and is on the verge of hanging herself, Len knocks on her door, having conquered his agoraphobia to alert her of Max's package. Inside, she finds the Noblet figurines and a letter from Max, in which he tells her of his realization that they are not perfect and expresses his forgiveness. He also states how much their friendship means to him, and that he hopes their paths will cross one day.\nOne year later, Mary travels to New York with her infant child to finally visit Max. Entering his apartment, Mary discovers Max on his couch, gazing upward with a smile on his face, having died earlier that morning. Looking around the apartment, Mary is awestruck to find all the letters she had sent to Max over the years, laminated and taped to the ceiling. Realizing Max had been gazing at the letters when he died, and seeing how much he had valued their friendship, Mary cries tears of joy and joins him on the couch.",
            "title":"Mary and Max",
            "question_id":"1dc019ad-80cf-1d49-5a69-368f90fae2f8",
            "question":"Why was Mary Daisy Dinkle teased in school?",
            "answers":[
               
            ],
            "no_answer":true
         },
         "truncated_cells":[
            
         ]
      }
   ],
   "num_rows_total":627,
   "num_rows_per_page":100,
   "partial":false
}
```

If the result has `partial: true` it means that the filtering couldn't be run on the full dataset because it's too big.

Indeed, the indexing for `/filter` can be partial if the dataset is bigger than 5GB. In that case, it only uses the first 5GB.
