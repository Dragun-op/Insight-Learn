# 📘 InsightLearn – Detecting True Understanding

**InsightLearn** is an AI-powered EdTech tool that analyzes student explanations to determine if they truly understand a topic or are merely memorizing. It uses NLP via Hugging Face APIs to classify responses as **Understood**, **Memorized**, or **Confused** — helping students self-evaluate and educators gain insight into learner comprehension.

## 🔧 Tech Stack

- **Backend**: Flask (Python), Flask-WTF, Flask-Login, Flask-Mail
- **Frontend**: HTML, CSS, Bootstrap 4 , JavaScript, Jinja2
- **Database**: PostgreSQL via SQLAlchemy
- **ML API**: Hugging Face Transformers (Zero-Shot Classification via API)

## 🧱 Folder Structure

```
insightlearn/
├── app/
│   ├── auth/                    # Authentication Blueprint (login, register, reset)
│   ├── nlp/                     # NLP Blueprint (text classification)
│   ├── static/                  # static files
│   ├── templates/
│   │   ├── auth/                # Auth pages
│   │   ├── nlp/                 # NLP pages (index, result)
│   │   ├── partials/            # Navbar, sidebar, footer
│   │   └── base.html            # Common layout
│   ├── models.py                # SQLAlchemy models
│   ├── forms.py                 # WTForms for auth
│   └── __init__.py              # Flask app factory
├── run.py                       # Entry point
├── .env                         # Environment variables
├── .flaskenv                    # Flask environment config
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

## 🚀 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/insightlearn.git
cd insightlearn
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 3. Configure environment variables in `.env`
```ini
SECRET_KEY=your_secret_key
DATABASE_URL=postgresql://username:password@localhost:5432/insightlearn
HF_API_KEY=your_huggingface_api_key
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=your_email
MAIL_PASSWORD=your_email_password
MAIL_DEFAULT_SENDER=your_email
```

### 4. Initialize the database
```bash
flask shell
>>> from app import db
>>> db.create_all()
>>> exit()
```

### 5. Run the application
```bash
flask run
```

The application will be available at `http://localhost:5000`

## 🚦 Feature Rollout Plan

### ✅ Phase 1: Core MVP Goals
- [x] User Authentication (Register, Login, Password Reset via Email)
- [x] Text-based explanation analysis using Hugging Face zero-shot classification
- [x] Classification into: Understood, Memorized, Confused
- [x] AdminLTE UI integration (Responsive layout)

### 📊 Phase 2: Data Expansion + History
- [ ] Save all explanation attempts in DB with timestamp
- [ ] Display user progress history via charts
- [ ] Show confidence scores and NLP-based recommendations
- [ ] Subject-specific prompt suggestions for better input

### 🧠 Phase 3: Multimodal Inputs
- [ ] Voice Input: Transcribe spoken explanation using WebRTC or browser speech recognition
- [ ] Facial Emotion Analysis via webcam to infer hesitation or confusion
- [ ] Fuse text + tone + expression for better classification accuracy

### 👩‍🏫 Phase 4: Educator Dashboard
- [ ] Admin view of user statistics and comprehension
- [ ] Flag students needing intervention
- [ ] Weak-topic detection and progress tracking
- [ ] Exportable reports for individual learners

## 🌱 Future Improvements

- Gamification for consistent engagement
- Real-time chat feedback from AI tutor
- Offline export of session reports (PDF/CSV)
- Adaptive content based on misunderstanding patterns

## 📋 Requirements

- Python 3.8+
- PostgreSQL 12+
- Hugging Face API Key
- Email account for SMTP (Gmail recommended)

## 🔑 API Configuration

### Hugging Face Setup
1. Create account at [Hugging Face](https://huggingface.co/)
2. Generate API key from Settings → Access Tokens
3. Add to `.env` as `HF_API_KEY`

### Email Configuration
For Gmail SMTP:
1. Enable 2-factor authentication
2. Generate app-specific password
3. Use app password as `MAIL_PASSWORD` in `.env`

## 🐛 Troubleshooting

### Common Issues

**Database Connection Error**
```bash
# Check PostgreSQL service is running
# Verify DATABASE_URL format
# Ensure database exists
```

**Hugging Face API Error**
```bash
# Verify API key is correct
# Check internet connection
# Ensure API quota not exceeded
```

**Email Not Sending**
```bash
# Verify SMTP settings
# Check email credentials
# Ensure less secure apps enabled (Gmail)
```

## 🤝 Contributing

Pull requests and suggestions are welcome! For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request