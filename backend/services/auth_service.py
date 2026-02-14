import jwt
import datetime
from flask import current_app

class AuthService:
    @staticmethod
    def generate_token(user_id):
        """
        Generate JWT token for a user.
        """
        try:
            payload = {
                'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1),
                'iat': datetime.datetime.utcnow(),
                'sub': user_id
            }
            return jwt.encode(
                payload,
                current_app.config.get('SECRET_KEY', 'default-secret'),
                algorithm='HS256'
            )
        except Exception as e:
            return str(e)

    @staticmethod
    def verify_token(token):
        """
        Verify JWT token.
        """
        try:
            payload = jwt.decode(
                token,
                current_app.config.get('SECRET_KEY', 'default-secret'),
                algorithms=['HS256']
            )
            return payload['sub']
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
