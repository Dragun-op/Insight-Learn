from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app.models import db, User, PasswordResetToken
from app.forms import LoginForm, RegisterForm, ForgotPasswordForm, ResetPasswordForm
from flask_mail import Message
from app import mail
import secrets
from datetime import datetime, timedelta

auth = Blueprint("auth", __name__)

@auth.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("nlp.index"))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for("nlp.index"))
        else:
            flash("Invalid email or password", "danger")
    return render_template("auth/login.html", form=form)

@auth.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("auth.login"))
    
    return render_template("auth/register.html", form=form)

@auth.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for("auth.login"))

@auth.route("/forgot", methods=["GET", "POST"])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            token = secrets.token_urlsafe(32)

            PasswordResetToken.query.filter_by(user_id=user.id).delete()

            reset_token = PasswordResetToken(
                token=token,
                user_id=user.id,
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )
            db.session.add(reset_token)
            db.session.commit()

            reset_link = url_for("auth.reset_password", token=token, _external=True)

            msg = Message("Reset Your Password", recipients=[user.email])
            msg.body = f"Click the link to reset your password: {reset_link}"
            mail.send(msg)

            flash("A reset link has been sent to your email.", "info")
        else:
            flash("No account found with that email.", "danger")
    return render_template("auth/forgot_password.html", form=form)

@auth.route("/reset/<token>", methods=["GET", "POST"])
def reset_password(token):
    form = ResetPasswordForm()

    reset_token = PasswordResetToken.query.filter_by(token=token).first()

    if not reset_token or reset_token.is_expired():
        flash("The reset link is invalid or has expired.", "danger")
        return redirect(url_for("auth.forgot_password"))

    user = reset_token.user

    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()

        db.session.delete(reset_token)
        db.session.commit()

        flash("Password reset successful. Please log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/reset_password.html", form=form)
