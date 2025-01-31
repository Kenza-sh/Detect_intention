from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import azure.functions as func
import logging
import re
class IntentionDetector:
    def __init__(self):
        # Define the phrases for training
        # Define the phrases to be used for training the model
        self.phrases_renseigner = [
            "Je voudrais des informations sur le cabinet",
            "Pouvez-vous me renseigner sur vos services?",
            "Quels sont les domaines d'expertise du cabinet?",
            "J'aimerais en savoir plus sur le cabinet",
            "Quelles informations pouvez-vous me donner?",
            "Quelles sont les heures d'ouverture?",
            "Où se situe votre cabinet?",
            "Comment fonctionne votre cabinet?",
            "Quels sont les moyens de paiement acceptés par le cabinet ?",
            "Acceptez-vous les cartes bancaires, ou seulement les chèques ?",
            "Puis-je régler mes consultations par virement bancaire ou PayPal ?",
            "Est-ce que vous acceptez les paiements en ligne ?",
            "Quels types de paiements sont possibles pour une consultation ?",
            "Acceptez-vous les assurances ou les mutuelles pour couvrir les frais ?",
            "Quels sont vos services?",
            "Quel est votre domaine de spécialisation?",
            "Est-ce que le cabinet prend en charge de nouveaux clients?",
            "Quels types de consultation proposez-vous?",
            "Quels sont les honoraires?",
            "Pouvez-vous m'informer sur les tarifs?",
            "Est-ce que le cabinet est ouvert le week-end?",
            "Avez-vous un site web où je peux consulter vos services?",
            "Comment puis-je contacter le cabinet?",
            "Quels sont vos horaires d'ouverture?",
            "Quels sont les moyens de communication avec le cabinet?",
            "Le cabinet propose-t-il des consultations à distance?",
            "Est-ce que vous acceptez la carte vitale?",
            "Quels sont les délais d'attente pour une consultation?",
            "Comment est-ce que je peux obtenir plus d'informations?",
            "Pouvez-vous m'envoyer des détails sur vos services?",
            "Quelle est votre adresse?",
            "Quelles sont les qualifications des praticiens?",
            "Pouvez-vous me donner des précisions sur vos offres?",
            "Est-ce que des consultations d'urgence sont disponibles?",
            "Combien de temps dure une consultation en moyenne?",
            "Avez-vous des services de suivi?",
            "Est-ce que vous faites des consultations à domicile?",
            "Pourriez-vous me fournir des informations complètes concernant votre cabinet et ses services ?",
            "J'aimerais obtenir plus de renseignements sur le cabinet et ses horaires, ainsi que les services proposés.",
            "Quelles sont les spécialités de votre cabinet et comment puis-je en savoir plus ?",
            "Est-ce que vous avez des informations sur les consultations à distance et les tarifs associés ?",
            "Je suis intéressé par des détails concernant vos offres, pourriez-vous me donner plus de précisions ?",
            "Pourriez-vous m'indiquer où je pourrais trouver plus d'informations sur le cabinet ?",
            "Je cherche à en savoir davantage sur vos horaires et services, auriez-vous des précisions à me donner ?",
            "Le cabinet est-il ouvert pendant les jours fériés ?",
            "Est-ce que vous travaillez lors des jours fériés, notamment à Noël ?",
            "Quelles sont vos heures d'ouverture pendant les vacances et jours fériés ?",
            "Est-ce que vous avez un planning spécial pour les jours fériés ?",
            "Le cabinet reste-t-il fermé pendant les jours fériés ou il y a-t-il une permanence ?",
            "Est-ce que vous êtes ouverts lors du Nouvel An ?",
            "Quel est le tarif moyen pour une consultation ?",
            "Pouvez-vous me donner une estimation des tarifs pour une première consultation ?",
            "Est-ce que vous avez des tarifs différents pour les consultations de suivi ?",
            "Quels sont vos tarifs pour les consultations en ligne ?",
            "Y a-t-il des frais supplémentaires pour les consultations urgentes ?",
            "Je voudrais connaître vos tarifs avant de prendre rendez-vous.",
            "Avez-vous une grille tarifaire que je pourrais consulter ?",
            "Quel est votre numéro de téléphone pour prendre rendez-vous ?",
            "Comment puis-je vous contacter par téléphone pour poser des questions ?",
            "Pouvez-vous me communiquer le numéro de contact pour le cabinet ?",
            "Je souhaite joindre votre cabinet par téléphone, quel est le numéro ?",
            "Comment puis-je obtenir un numéro pour vous contacter directement ?",
            "Est-ce que le cabinet a un numéro de contact spécifique pour les urgences ?",
            "J'aimerais discuter avec un praticien, quel est le moyen de vous joindre ?"
            "Comment prendre rendez-vous ?",
            "Dois-je apporter mes anciens examens ?",
            "Est-ce que la mammographie fait mal ?",
            "Quels documents dois-je apporter pour mon examen ?",
            "Je pense être enceinte, puis-je passer mon examen ?"
            "Est-ce que je dois attendre en salle d'attente le compte rendu du radiologue ?",
            "Dois-je apporter mes anciennes radios à ma consultation ?",
            "Faut-il être à jeun pour ma prise de sang de demain ?",
            "Puis-je venir accompagné pour mon coloscan ?",
            "Comment obtenir un justificatif pour mon employeur ?",
            "Mon ordonnance a expiré, puis-je quand même venir ?",
             "Quelles mutuelles acceptez-vous pour les séances de kiné ?",
            "Faut-il une ordonnance pour une radiographie pulmonaire ?",
            "Délai moyen pour obtenir les résultats d'une biopsie ?",
            "Puis-je avoir un deuxième avis sur mon IRM ?",
            "Vos locaux sont-ils adaptés aux fauteuils roulants ?",
            "Que faire en cas de perte de mes résultats d'analyse ?",
            "Procédure pour crise d'épilepsie pendant une consultation",
            "Service de garde pour les week-ends et jours fériés ?",
            "Comment contacter le radiologue en urgence la nuit ?",
        ]

        self.phrases_gerer_rdv = [
            "Gestion des rendez-vous",
            "Annulation de rendez-vous",
            "Modification de rendez-vous",
            "Prise de rendez-vous",
            "Plannification",
            "Réservation ",
            "je souhaite gérer mes rendez-vous",
            "Je veux prendre un rendez-vous",
            "Puis-je modifier mon rendez-vous?",
            "J'aimerais annuler mon rendez-vous",
            "Aidez-moi à gérer mes rendez-vous",
            "Comment puis-je organiser un rendez-vous?",
            "Puis-je déplacer mon rendez-vous?",
            "Je souhaite planifier un rendez-vous",
            "Comment réserver un rendez-vous?",
            "Je voudrais confirmer mon rendez-vous",
            "Pouvez-vous m'aider à programmer un rendez-vous?",
            "Est-il possible de reporter mon rendez-vous?",
            "Je voudrais vérifier la date de mon rendez-vous",
            "J'ai besoin de changer l'horaire de mon rendez-vous",
            "Pouvez-vous me rappeler la date de mon rendez-vous?",
            "Est-il possible de prendre rendez-vous en ligne?",
            "Comment puis-je contacter pour un changement de rendez-vous?",
            "Puis-je reprogrammer ma consultation?",
            "Je veux fixer une nouvelle date de rendez-vous",
            "Puis-je planifier plusieurs rendez-vous?",
            "J'ai besoin de décaler mon rendez-vous",
            "Comment puis-je annuler ma consultation?",
            "Avez-vous des créneaux disponibles?",
            "Est-il possible de confirmer mon rendez-vous?",
            "Je souhaite ajuster l'heure de mon rendez-vous",
            "Est-ce que je peux fixer une consultation cette semaine?",
            "Puis-je réserver une consultation pour un proche?",
            "Comment procéder pour réserver un rendez-vous?",
            "Je dois m'inscrire pour une consultation",
            "Est-ce que le cabinet accepte les rendez-vous urgents?",
            "Avez-vous des créneaux libres dans les prochains jours?",
            "Puis-je planifier une visite de suivi?",
            "Quel est le délai pour obtenir un rendez-vous?",
            "Je voudrais vérifier la disponibilité pour un rendez-vous",
            "Je cherche un rendez-vous le plus tôt possible",
            "Est-ce que je peux annuler ou changer un rendez-vous en ligne?",
            "Puis-je organiser un rendez-vous pour plusieurs personnes?",
            "Quel est le processus pour annuler un rendez-vous?",
            "J'aimerais reprogrammer un rendez-vous pour la semaine prochaine, est-ce possible ?",
            "Pouvez-vous m'aider à annuler ma consultation prévue pour demain et la replanifier à une autre date ?",
            "Je souhaite savoir s'il est possible de modifier l'heure de mon rendez-vous, j'ai un conflit d'agenda.",
            "J'aimerais réserver un créneau pour une consultation, quelles sont vos disponibilités cette semaine ?",
            "Est-ce qu'il y a une manière de confirmer ma réservation de rendez-vous en ligne sans passer par l'accueil ?",
            "Est-ce que je peux reporter mon rendez-vous de jeudi à vendredi sans perdre ma place ?",
            "Est-il possible de changer mon rendez-vous, si oui, comment faire ?",
            "J'aimerais vérifier l'heure exacte de ma consultation prévue, est-ce que vous pouvez m'aider ?",
            "Je voudrais annuler mon rendez-vous, mais je ne suis pas sûr de comment procéder, pouvez-vous m'aider ?",
            "Annuler mon RDV suite à amélioration",
            "Mon médecin m'a prescrit un autre examen, je dois annuler mon scanner",
            "Je dois annuler ma consultation gynécologique pour cause d'hospitalisation",
            "Procédure d'annulation pour une échographie Doppler",
            "je dois avancer mon RDV ",
            "Comment déplacer mon bilan prévu cet après-midi ?",
            "Puis-je changer de rendez vous pour mon IRM cérébral ?",
            "Je souhaite ajouter une échographie à mon examen de demain",
            
            
        ]
        # Combine all phrases
        self.all_phrases = self.phrases_renseigner + self.phrases_gerer_rdv
        
        # Create TF-IDF vectorizer and fit on phrases
        self.vectorizer = TfidfVectorizer()
        self.phrase_vectors = self.vectorizer.fit_transform(self.all_phrases)

    def detect_intention(self, reponse):
        if not isinstance(reponse, str) or not reponse.strip():
            return "renseigner"
        
        # Transform the user's response
        user_vector = self.vectorizer.transform([reponse])
        
        # Compute cosine similarity
        similarities = cosine_similarity(user_vector, self.phrase_vectors)
        best_index = similarities.argmax()
        
        # Determine the intention
        if best_index < len(self.phrases_renseigner):
            return "renseigner"
        else:
            return "gérer"

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        query = req_body.get('text')

        if not query:
            return func.HttpResponse(
                json.dumps({"error": "No query provided in request body"}),
                mimetype="application/json",
                status_code=400
            )
        detector = IntentionDetector()
        result = detector.detect_intention(query)

        return func.HttpResponse(
            json.dumps({"response": result}),
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )
