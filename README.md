
# Hybrid Recommendation System

This recommendation system is backed by two **rule-based systems** to avoid the cold-start problem. Additionally, synthetic data is being created following patterns that align with real-world scenarios.

This project is a **hybrid recommendation system** developed to address a real-world problem. The goal is to recommend resources to women and their partners during the perinatal period, aiming to prevent mental health disorders such as depression or anxiety.

**Project Status**:  
🛠️ **In progress**  
The system is under development, and some functionalities may not be fully implemented or optimized. Continuous improvements and documentation will be added over time.

## Real-World Problem

The project is based on a real situation: the perinatal period is a critical stage for the mental health of women and their partners. The system seeks to offer personalized recommendations for interventions and resources to help prevent issues such as postpartum depression or anxiety.

### Main Challenge

One of the main challenges of this project is the lack of real user data. Due to the nature of the problem, there is no historical data available to train the recommendation algorithm. This aspect adds complexity to the training and evaluation process.

## Implicit and Explicit Interactions

The system handles both **implicit interactions** (such as browsing behavior or saved resources) and **explicit interactions** (direct user ratings). This dual approach adds more complexity to the algorithm but also improves the accuracy of the recommendations.

## Implemented Features

- **Content-Based Recommendations**: Personalized suggestions based on the interests and behaviors of users and their partners.
- **User-Based Collaborative Filtering**: Recommendations based on similarities between users.
- **Item-Based Collaborative Filtering**: Recommendations based on item similarity.
- **Hybrid Recommendations**: Combination of the above approaches with adjustable weights.

## Features to be Implemented

- Optimization of hybrid recommendation weights.
- Advanced system evaluation (precision, recall, F1, diversity, novelty).
- Expansion of the recommended resource base.

## Requirements

Currently, there is no `requirements.txt` file. The main dependencies include libraries such as `pandas`, `numpy`, `psycopg2`, and `scikit-learn`.

## Usage

The system currently provides personalized recommendations based on user characteristics and behaviors. To run the system:

```bash
python hybrid_sistem_recommendation.py
```

## Roadmap

- Implement advanced evaluation metrics.
- Improve user interface and API documentation.
- Expand recommendation methods to increase accuracy.
