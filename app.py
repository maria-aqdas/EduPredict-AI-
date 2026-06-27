import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def generate_mock_student_data():
    """Generates synthetic historical student data for training."""
    np.random.seed(42)
    n_students = 400
    
    # Independent variables/Features
    attendance = np.random.randint(60, 100, size=n_students)
    quiz_scores = np.random.randint(40, 100, size=n_students)
    assignment_completion = np.random.randint(50, 100, size=n_students)
    study_hours_weekly = np.random.randint(2, 20, size=n_students)
    
    # Logic to create a target variable (Pass/Fail) based on features
    # Passing = 1, Failing/At Risk = 0
    score_formula = (attendance * 0.3) + (quiz_scores * 0.4) + (assignment_completion * 0.2) + (study_hours_weekly * 1.0)
    performance_status = np.where(score_formula > 65, 1, 0)
    
    df = pd.DataFrame({
        'Attendance': attendance,
        'QuizScores': quiz_scores,
        'AssignmentCompletion': assignment_completion,
        'StudyHours': study_hours_weekly,
        'Status': performance_status
    })
    return df

def get_personalized_roadmap(attendance, quiz_scores, prediction):
    """Generates an automated study plan baseline."""
    roadmap = []
    if prediction == 0:
        roadmap.append("🚨 STATUS: At Academic Risk. Core attention required immediately.")
    else:
        roadmap.append("✨ STATUS: Good Standing. Maintenance roadmap active.")
        
    if attendance < 80:
        roadmap.append("- Action 1: Attend peer mentoring sessions and clear outstanding lecture gaps.")
    if quiz_scores < 70:
        roadmap.append("- Action 2: Review foundational problem sets. Focus heavily on practice exams.")
    if len(roadmap) == 1:
        roadmap.append("- Action: Continue with current study schedule. Focus on advanced elective topics.")
        
    return "\n".join(roadmap)

print("--- EduPredict AI: Running Core Pipeline ---")

# 1. Load Data
data = generate_mock_student_data()
X = data[['Attendance', 'QuizScores', 'AssignmentCompletion', 'StudyHours']]
y = data['Status']

# 2. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate Pipeline
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Training Complete. Pipeline Accuracy Evaluation: {accuracy * 100:.2f}%\n")

# 4. Simulating live prediction & tailored roadmap generation
print("--- Simulating Live Prediction Output (Sample Student) ---")

# Sample student data wrapped in a DataFrame with exact feature names to prevent warnings
sample_student_metrics = pd.DataFrame(
    [[75, 55, 60, 4]], 
    columns=['Attendance', 'QuizScores', 'AssignmentCompletion', 'StudyHours']
)

predicted_outcome = model.predict(sample_student_metrics)[0]

generated_roadmap = get_personalized_roadmap(
    attendance=sample_student_metrics.iloc[0]['Attendance'], 
    quiz_scores=sample_student_metrics.iloc[0]['QuizScores'], 
    prediction=predicted_outcome
)

print(f"Input Metrics -> Attendance: {sample_student_metrics.iloc[0]['Attendance']}%, Quiz Avg: {sample_student_metrics.iloc[0]['QuizScores']}%")
print("\nGenerated Academic Roadmap:\n" + generated_roadmap)
print("----------------------------------------------------------")
