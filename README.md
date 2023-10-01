# Named-Entity Recognition Using CRF and BERT

Named-entity recognition (NER) is a natural language processing technique. It is also called *entity identification* or *entity extraction*. It identifies named entities in text and classifies them into predefined categories. For example, extracted entities can be the names of organizations, locations, times, quantities, people, monetary values, and more present in text.

With NER, key information is often extracted to learn what a given text is about, or it is used to gather important information to store in a database.

NER is used in many applications across many domains. NER is extensively used in biomedical data. For instance, it is used for DNA identification, gene identification, and the identification of drug names and disease names. Figure 1 shows an example of a medical text-related NER that extracts symptoms, patient type, and dosage.

![IMAGE_GIT](https://github.com/sohansai/named-entity-recognition/assets/76840110/ff42aac1-ec61-4f95-8381-e98eaaef452e)

NER is also used for optimizing the search queries and ranking search results. It is sometimes combined with topic identification. NER is also used in machine translation.

There are a lot of pretrained general-purpose libraries that use NER. For example, spaCy-an open source Python library for various NLP tasks. And NLTK (natural language tool kit) has a wrapper for the Stanford NER, which is simpler in many cases.

These libraries only extract a certain type of entities, like name, location, and so on. If you need to extract something very domain-specific, such as the name of a medical treatment, it is impossible. You need to build custom NER in those scenarios.
