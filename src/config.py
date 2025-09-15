"""Configuration management for the Spend Platform application."""
import os
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    path: str
    max_connections: int = 5
    timeout: int = 30


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    secret_key: str
    session_timeout: timedelta
    password_min_length: int = 8
    max_login_attempts: int = 3


@dataclass
class UploadConfig:
    """File upload configuration settings."""
    max_file_size: int
    allowed_extensions: List[str]
    upload_folder: Path


@dataclass
class UIConfig:
    """User interface configuration settings."""
    default_chart_height: int
    records_per_page: List[int]
    recent_items_limit: int


class Config:
    """Application configuration manager."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables and defaults."""
        self.env = os.getenv('FLASK_ENV', 'development')
        
        # Application settings
        self.APP_TITLE = "Spend Data Management Platform"
        self.APP_ICON = "💰"
        
        # Database settings
        self.database = DatabaseConfig(
            path=os.getenv('DB_PATH', 'spend_platform.db'),
            max_connections=int(os.getenv('DB_MAX_CONNECTIONS', '5')),
            timeout=int(os.getenv('DB_TIMEOUT', '30'))
        )
        
        # Security settings
        self.security = SecurityConfig(
            secret_key=os.getenv('SECRET_KEY', 'dev-key-change-in-production'),
            session_timeout=timedelta(hours=int(os.getenv('SESSION_TIMEOUT_HOURS', '8'))),
            password_min_length=int(os.getenv('PASSWORD_MIN_LENGTH', '8')),
            max_login_attempts=int(os.getenv('MAX_LOGIN_ATTEMPTS', '3'))
        )
        
        # Upload settings
        self.upload = UploadConfig(
            max_file_size=int(os.getenv('MAX_FILE_SIZE', str(100 * 1024 * 1024))),  # 100MB
            allowed_extensions=['.xlsx', '.xls', '.csv'],
            upload_folder=Path(os.getenv('UPLOAD_FOLDER', 'uploads'))
        )
        
        # UI settings
        self.ui = UIConfig(
            default_chart_height=int(os.getenv('DEFAULT_CHART_HEIGHT', '400')),
            records_per_page=[10, 15, 25, 50, 100],
            recent_items_limit=int(os.getenv('RECENT_ITEMS_LIMIT', '10'))
        )
        
        # Error types
        self.error_types = [
            'Negative Amount',
            'Missing Supplier',
            'Invalid Date',
            'Duplicate Transaction',
            'Missing Category',
            'Invalid Currency',
            'Missing BU Code',
            'Data Quality Issue'
        ]
        
        # User roles and permissions
        self.roles: Dict[str, Dict[str, Any]] = {
            'Admin': {
                'can_manage_users': True,
                'can_manage_master_data': True,
                'can_upload_data': True,
                'can_manage_rules': True,
                'can_view_reports': True,
                'can_resolve_errors': True
            },
            'Spend Manager': {
                'can_manage_users': False,
                'can_manage_master_data': True,
                'can_upload_data': True,
                'can_manage_rules': False,
                'can_view_reports': True,
                'can_resolve_errors': True
            },
            'Data Analyst': {
                'can_manage_users': False,
                'can_manage_master_data': False,
                'can_upload_data': False,
                'can_manage_rules': False,
                'can_view_reports': True,
                'can_resolve_errors': False
            }
        }

    def get_db_uri(self) -> str:
        """Get the database URI."""
        return f"sqlite:///{self.database.path}"

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env == 'production'

    def validate(self) -> None:
        """Validate the configuration settings."""
        if self.is_production():
            assert self.security.secret_key != 'dev-key-change-in-production', \
                "Production requires a proper secret key"
            assert self.database.path != ':memory:', \
                "Production requires a proper database path"


# Global config instance
config = Config()
