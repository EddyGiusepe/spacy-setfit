"""
Data Scientist.: Dr.Eddy Giusepe Chirinos Iisdro

Objetivo: Aqui seguimos o repositório de "David Berenstein" para 
          usar spaCy com SetFit. O exemplo neste script retorna 
          as categorias previstas e suas probabilidades associadas.
          
                 
Link de estudo: https://github.com/davidberenstein1957/spacy-setfit
"""
import spacy

# Crie alguns dados de exemplo:
train_dataset = {
    "inlier": ["Este texto é sobre cadeiras.",
               "Sofás, bancos e televisores.",
               "Preciso muito de um sofá novo."],
    "outlier": ["Texto sobre equipamentos de cozinha",
                "Este texto é sobre política",
                "Comentários sobre IA e outras coisas."]
}

# Carregamos o modelo pré-treinado do spaCy:
nlp = spacy.load("en_core_web_sm")

# Adicione o componente de pipeline "text_categorizer" ao modelo spaCy e
# configure-o com os parâmetros SetFit:
nlp.add_pipe("text_categorizer", config={
    "pretrained_model_name_or_path": "sentence-transformers/paraphrase-MiniLM-L3-v2",
    "setfit_trainer_args": {
        "train_dataset": train_dataset
    }
})
doc = nlp("Preciso muito de um sofá novo.")
doc.cats
print(doc.cats)
# {'inlier': 0.9282628061922414, 'outlier': 0.07173719380775859}