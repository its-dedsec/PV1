import datetime
import json
import random
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# =============================================================================
# DATA STRUCTURES AND ENUMS
# =============================================================================

class VulnerabilitySeverity(Enum):
    """Enumeration for vulnerability severity levels."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    INFO = 1
    
    @classmethod
    def from_cvss(cls, cvss_score: float) -> 'VulnerabilitySeverity':
        """Convert CVSS score to severity enum."""
        if cvss_score >= 9.0:
            return cls.CRITICAL
        elif cvss_score >= 7.0:
            return cls.HIGH
        elif cvss_score >= 4.0:
            return cls.MEDIUM
        elif cvss_score >= 0.1:
            return cls.LOW
        else:
            return cls.INFO


class AssetCriticality(Enum):
    """Enumeration for asset criticality levels."""
    MISSION_CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1


class VulnerabilityCategory(Enum):
    """Common vulnerability categories."""
    WEAK_AUTHENTICATION = "Weak Authentication"
    OUTDATED_SOFTWARE = "Outdated Software"
    MISCONFIGURATION = "System Misconfiguration"
    MISSING_PATCH = "Missing Security Patch"
    DEFAULT_CREDENTIALS = "Default Credentials"
    SQL_INJECTION = "SQL Injection"
    XSS = "Cross-Site Scripting"
    CSRF = "Cross-Site Request Forgery"
    OPEN_PORT = "Unnecessary Open Port"
    UNNECESSARY_SERVICE = "Unnecessary Service"
    INFORMATION_DISCLOSURE = "Information Disclosure"
    FILE_INCLUSION = "File Inclusion Vulnerability"


class ExploitStatus(Enum):
    """Exploit availability status."""
    PUBLIC = "Public exploit available"
    PRIVATE = "Private exploit likely available"
    POC = "Proof of concept available"
    THEORETICAL = "Theoretical exploit"
    NONE = "No known exploit"


class Asset:
    """Represents a network asset."""
    
    def __init__(
        self,
        asset_id: str,
        hostname: str,
        ip_address: str,
        os: str,
        criticality: AssetCriticality,
        owner: str,
        department: str,
        services: List[Dict[str, Union[int, str]]] = None
    ):
        self.asset_id = asset_id
        self.hostname = hostname
        self.ip_address = ip_address
        self.os = os
        self.criticality = criticality
        self.owner = owner
        self.department = department
        self.services = services or []
        self.vulnerabilities = []
    
    def to_dict(self) -> Dict:
        """Convert asset to dictionary."""
        return {
            "asset_id": self.asset_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "os": self.os,
            "criticality": self.criticality.name,
            "owner": self.owner,
            "department": self.department,
            "services": self.services,
            "vulnerabilities_count": len(self.vulnerabilities),
        }
    
    def __str__(self) -> str:
        return f"Asset({self.hostname}, {self.ip_address}, {self.criticality.name})"


class Vulnerability:
    """Represents a detected vulnerability."""
    
    def __init__(
        self,
        vuln_id: str,
        name: str,
        description: str,
        cvss_score: float,
        category: VulnerabilityCategory,
        affected_asset: Asset,
        detection_date: datetime.datetime,
        cve_id: Optional[str] = None,
        exploit_status: Optional[ExploitStatus] = None,
        remediation_steps: Optional[List[str]] = None,
        scanner_source: Optional[str] = None,
    ):
        self.vuln_id = vuln_id
        self.name = name
        self.description = description
        self.cvss_score = cvss_score
        self.severity = VulnerabilitySeverity.from_cvss(cvss_score)
        self.category = category
        self.affected_asset = affected_asset
        self.detection_date = detection_date
        self.cve_id = cve_id
        self.exploit_status = exploit_status or self._generate_exploit_status()
        self.remediation_steps = remediation_steps
        self.scanner_source = scanner_source
        self.risk_score = None  # Will be calculated by scoring engine
        self.validation_status = None  # Will be set by validation engine
        
        # Add vulnerability to the asset
        affected_asset.vulnerabilities.append(self)
    
    def _generate_exploit_status(self) -> ExploitStatus:
        """Generate a realistic exploit status based on CVSS score."""
        if self.cvss_score >= 9.0:
            weights = [0.6, 0.2, 0.1, 0.1, 0.0]
        elif self.cvss_score >= 7.0:
            weights = [0.3, 0.3, 0.2, 0.1, 0.1]
        elif self.cvss_score >= 4.0:
            weights = [0.1, 0.2, 0.3, 0.3, 0.1]
        else:
            weights = [0.0, 0.1, 0.2, 0.2, 0.5]
        
        return random.choices(list(ExploitStatus), weights=weights)[0]
    
    def to_dict(self) -> Dict:
        """Convert vulnerability to dictionary."""
        return {
            "vuln_id": self.vuln_id,
            "name": self.name,
            "description": self.description,
            "cvss_score": self.cvss_score,
            "severity": self.severity.name,
            "category": self.category.name,
            "affected_asset": {
                "asset_id": self.affected_asset.asset_id,
                "hostname": self.affected_asset.hostname,
                "ip_address": self.affected_asset.ip_address
            },
            "detection_date": self.detection_date.isoformat(),
            "cve_id": self.cve_id,
            "exploit_status": self.exploit_status.name if self.exploit_status else None,
            "risk_score": self.risk_score,
            "validation_status": self.validation_status,
            "scanner_source": self.scanner_source,
        }
    
    def __str__(self) -> str:
        return f"Vulnerability({self.name}, CVSS: {self.cvss_score}, {self.severity.name})"


# =============================================================================
# REMEDIATION DATABASE
# =============================================================================

REMEDIATION_DATABASE = {
    VulnerabilityCategory.WEAK_AUTHENTICATION: [
        "Implement multi-factor authentication",
        "Enforce password complexity requirements",
        "Implement account lockout policies",
        "Use password managers and unique passwords for each service",
        "Regular password rotation for critical systems"
    ],
    VulnerabilityCategory.OUTDATED_SOFTWARE: [
        "Update software to the latest stable version",
        "Implement automated patch management system",
        "Create a software inventory and update policy",
        "Set up vulnerability notifications for installed software",
        "Test updates in staging environment before production deployment"
    ],
    VulnerabilityCategory.MISCONFIGURATION: [
        "Apply security benchmarks and hardening guidelines",
        "Use configuration management tools",
        "Implement regular configuration audits",
        "Remove unnecessary features and services",
        "Apply the principle of least privilege"
    ],
    VulnerabilityCategory.MISSING_PATCH: [
        "Apply security patches immediately for critical vulnerabilities",
        "Implement automated patch management",
        "Create and follow a patch management policy",
        "Test patches in non-production environment first",
        "Maintain an up-to-date inventory of systems requiring patches"
    ],
    VulnerabilityCategory.DEFAULT_CREDENTIALS: [
        "Change all default credentials immediately",
        "Implement credential management system",
        "Perform regular audits for default credentials",
        "Create unique credentials for each system",
        "Document credential changes in secure location"
    ],
    VulnerabilityCategory.SQL_INJECTION: [
        "Use parameterized queries or prepared statements",
        "Implement input validation and sanitization",
        "Apply the principle of least privilege to database accounts",
        "Use ORM frameworks with built-in protection",
        "Implement web application firewall (WAF)"
    ],
    VulnerabilityCategory.XSS: [
        "Implement Content Security Policy (CSP)",
        "Sanitize and validate all user inputs",
        "Use context-aware escaping for output",
        "Use modern frameworks with built-in XSS protection",
        "Implement XSS filters and WAF rules"
    ],
    VulnerabilityCategory.CSRF: [
        "Implement anti-CSRF tokens",
        "Use SameSite cookie attribute",
        "Verify origin and referrer headers",
        "Implement proper session management",
        "Use framework-provided CSRF protection mechanisms"
    ],
    VulnerabilityCategory.OPEN_PORT: [
        "Close unnecessary ports",
        "Implement network segmentation",
        "Use firewalls to restrict access to necessary ports only",
        "Implement regular port scanning",
        "Document all required open ports and their purpose"
    ],
    VulnerabilityCategory.UNNECESSARY_SERVICE: [
        "Disable or uninstall unnecessary services",
        "Implement regular service audits",
        "Apply the principle of least functionality",
        "Document all required services and their purpose",
        "Use containerization to isolate services"
    ],
    VulnerabilityCategory.INFORMATION_DISCLOSURE: [
        "Configure proper error handling",
        "Remove sensitive information from responses",
        "Implement proper access controls",
        "Use secure headers",
        "Regularly audit for information leakage"
    ],
    VulnerabilityCategory.FILE_INCLUSION: [
        "Validate and sanitize file paths",
        "Implement proper access controls",
        "Use whitelisting for allowed files and directories",
        "Avoid passing user input to file system operations",
        "Implement proper error handling"
    ]
}


# =============================================================================
# SIMULATED DATA GENERATORS
# =============================================================================

class SimulatedDataGenerator:
    """Generate simulated assets and vulnerabilities for testing."""
    
    # Lists for generating realistic sample data
    OS_LIST = ["Windows Server 2019", "Windows Server 2016", "Ubuntu 20.04 LTS", 
               "CentOS 8", "Debian 10", "Red Hat Enterprise Linux 8", "macOS 11.0"]
    
    SERVICE_TYPES = ["HTTP", "HTTPS", "SSH", "FTP", "SMTP", "DNS", "SMB", 
                    "RDP", "Telnet", "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch"]
    
    DEPARTMENTS = ["IT", "Finance", "HR", "Engineering", "Marketing", "Sales", 
                  "Research", "Development", "Operations", "Executive", "Legal", "Support"]
    
    VULNERABILITY_NAMES = {
        VulnerabilityCategory.WEAK_AUTHENTICATION: [
            "Weak password policy", 
            "Lack of multi-factor authentication", 
            "Password stored in plaintext"
        ],
        VulnerabilityCategory.OUTDATED_SOFTWARE: [
            "Outdated web server version", 
            "Deprecated PHP version", 
            "Legacy operating system", 
            "End-of-life software"
        ],
        VulnerabilityCategory.MISCONFIGURATION: [
            "Default server configuration", 
            "Overly permissive file permissions", 
            "Excessive user privileges", 
            "Unnecessary service enabled"
        ],
        VulnerabilityCategory.MISSING_PATCH: [
            "Missing critical security patch", 
            "Unpatched kernel vulnerability", 
            "Security update not applied"
        ],
        VulnerabilityCategory.DEFAULT_CREDENTIALS: [
            "Default admin credentials", 
            "Factory password not changed", 
            "Default SSH keys"
        ],
        VulnerabilityCategory.SQL_INJECTION: [
            "Blind SQL Injection", 
            "Error-based SQL Injection", 
            "Time-based SQL Injection"
        ],
        VulnerabilityCategory.XSS: [
            "Reflected XSS", 
            "Stored XSS", 
            "DOM-based XSS"
        ],
        VulnerabilityCategory.CSRF: [
            "CSRF in form submission", 
            "CSRF token missing", 
            "CSRF in API endpoint"
        ],
        VulnerabilityCategory.OPEN_PORT: [
            "Unnecessary open port", 
            "Exposed debug port", 
            "Unfiltered network service"
        ],
        VulnerabilityCategory.UNNECESSARY_SERVICE: [
            "Deprecated service running", 
            "Unnecessary daemon", 
            "Unused service active"
        ],
        VulnerabilityCategory.INFORMATION_DISCLOSURE: [
            "Server version disclosure", 
            "Detailed error messages", 
            "Directory listing enabled"
        ],
        VulnerabilityCategory.FILE_INCLUSION: [
            "Local File Inclusion", 
            "Remote File Inclusion", 
            "Path Traversal"
        ]
    }
    
    SCANNER_NAMES = ["Nessus", "OpenVAS", "QualysGuard"]
    
    @classmethod
    def generate_assets(cls, count: int = 20) -> List[Asset]:
        """Generate a list of simulated assets."""
        assets = []
        
        for i in range(count):
            # Generate a valid IP address
            ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
            
            # Select OS, criticality and department
            os = random.choice(cls.OS_LIST)
            criticality = random.choice(list(AssetCriticality))
            department = random.choice(cls.DEPARTMENTS)
            
            # Generate hostname based on department and function
            hostname = f"{department.lower()}-{random.choice(['srv', 'app', 'db', 'web', 'sec'])}{i:02d}"
            
            # Generate some services
            services_count = random.randint(1, 5)
            services = []
            for _ in range(services_count):
                service_type = random.choice(cls.SERVICE_TYPES)
                port = cls._get_default_port_for_service(service_type)
                services.append({
                    "name": service_type,
                    "port": port,
                    "version": f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 20)}"
                })
            
            # Create the asset
            asset = Asset(
                asset_id=str(uuid.uuid4()),
                hostname=hostname,
                ip_address=ip,
                os=os,
                criticality=criticality,
                owner=f"user{random.randint(100, 999)}@example.com",
                department=department,
                services=services
            )
            assets.append(asset)
        
        return assets
    
    @classmethod
    def _get_default_port_for_service(cls, service_type: str) -> int:
        """Return default port for a given service type."""
        port_map = {
            "HTTP": 80,
            "HTTPS": 443,
            "SSH": 22,
            "FTP": 21,
            "SMTP": 25,
            "DNS": 53,
            "SMB": 445,
            "RDP": 3389,
            "Telnet": 23,
            "MySQL": 3306,
            "PostgreSQL": 5432,
            "MongoDB": 27017,
            "Redis": 6379,
            "Elasticsearch": 9200,
        }
        return port_map.get(service_type, random.randint(1000, 9999))
    
    @classmethod
    def generate_vulnerabilities(cls, assets: List[Asset], count_range: Tuple[int, int] = (0, 8)) -> List[Vulnerability]:
        """Generate vulnerabilities for given assets."""
        vulnerabilities = []
        
        # Decide how many vulnerabilities to generate for each asset
        for asset in assets:
            # Higher criticality assets tend to have more scrutiny and fewer vulnerabilities
            # Adjust the max number of vulnerabilities based on asset criticality
            if asset.criticality == AssetCriticality.MISSION_CRITICAL:
                max_vuln = 3
            elif asset.criticality == AssetCriticality.HIGH:
                max_vuln = 4
            elif asset.criticality == AssetCriticality.MEDIUM:
                max_vuln = 6
            else:
                max_vuln = 8
            
            # Generate between min and max vulnerabilities for this asset
            num_vulns = random.randint(count_range[0], min(count_range[1], max_vuln))
            
            # Generate each vulnerability
            for _ in range(num_vulns):
                # Select a random vulnerability category
                category = random.choice(list(VulnerabilityCategory))
                
                # Select a vulnerability name from that category
                name = random.choice(cls.VULNERABILITY_NAMES[category])
                
                # Generate a realistic CVSS score based on category and asset criticality
                base_cvss = cls._generate_cvss_for_category(category)
                criticality_adjustment = 0.5 * asset.criticality.value
                cvss_score = min(10.0, base_cvss + random.uniform(-0.5, 0.5) + criticality_adjustment)
                
                # Generate a CVE ID for some vulnerabilities
                cve_id = None
                if random.random() < 0.7:  # 70% chance of having a CVE
                    year = random.randint(2015, 2024)
                    cve_number = random.randint(1000, 99999)
                    cve_id = f"CVE-{year}-{cve_number}"
                
                # Create a description
                description = cls._generate_description(name, category, asset)
                
                # Select a random scanner as the source
                scanner_source = random.choice(cls.SCANNER_NAMES)
                
                # Create the vulnerability
                vulnerability = Vulnerability(
                    vuln_id=str(uuid.uuid4()),
                    name=name,
                    description=description,
                    cvss_score=cvss_score,
                    category=category,
                    affected_asset=asset,
                    detection_date=datetime.datetime.now() - datetime.timedelta(days=random.randint(0, 30)),
                    cve_id=cve_id,
                    scanner_source=scanner_source,
                    remediation_steps=REMEDIATION_DATABASE.get(category, ["No specific remediation steps available"])
                )
                
                vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    @classmethod
    def _generate_cvss_for_category(cls, category: VulnerabilityCategory) -> float:
        """Generate a realistic CVSS score based on vulnerability category."""
        category_cvss_ranges = {
            VulnerabilityCategory.WEAK_AUTHENTICATION: (5.0, 8.5),
            VulnerabilityCategory.OUTDATED_SOFTWARE: (4.0, 7.5),
            VulnerabilityCategory.MISCONFIGURATION: (3.0, 7.0),
            VulnerabilityCategory.MISSING_PATCH: (4.5, 9.0),
            VulnerabilityCategory.DEFAULT_CREDENTIALS: (5.5, 8.0),
            VulnerabilityCategory.SQL_INJECTION: (7.0, 9.5),
            VulnerabilityCategory.XSS: (4.5, 7.0),
            VulnerabilityCategory.CSRF: (4.0, 6.5),
            VulnerabilityCategory.OPEN_PORT: (2.5, 5.5),
            VulnerabilityCategory.UNNECESSARY_SERVICE: (2.0, 5.0),
            VulnerabilityCategory.INFORMATION_DISCLOSURE: (2.0, 4.5),
            VulnerabilityCategory.FILE_INCLUSION: (5.5, 8.5),
        }
        
        min_score, max_score = category_cvss_ranges.get(category, (2.0, 7.0))
        return round(random.uniform(min_score, max_score), 1)
    
    @classmethod
    def _generate_description(cls, name: str, category: VulnerabilityCategory, asset: Asset) -> str:
        """Generate a realistic vulnerability description."""
        # Base descriptions for each category
        descriptions = {
            VulnerabilityCategory.WEAK_AUTHENTICATION: 
                f"The {name} vulnerability was detected on {asset.hostname}. The system's authentication " +
                "mechanisms do not adequately protect against unauthorized access. ",
            
            VulnerabilityCategory.OUTDATED_SOFTWARE: 
                f"The system is running an outdated version of software which contains known security issues. " +
                f"The {name} vulnerability affects {random.choice([s['name'] for s in asset.services]) if asset.services else asset.os}. ",
            
            VulnerabilityCategory.MISCONFIGURATION: 
                f"A security misconfiguration was detected: {name}. This improper configuration can " +
                "lead to unauthorized access or information disclosure. ",
            
            VulnerabilityCategory.MISSING_PATCH: 
                f"The system is missing an important security patch. {name} leaves the system " +
                f"vulnerable to attacks targeting {asset.os}. ",
            
            VulnerabilityCategory.DEFAULT_CREDENTIALS: 
                f"{name} were detected on {asset.hostname}. Using default credentials significantly " +
                "increases the risk of unauthorized access. ",
            
            VulnerabilityCategory.SQL_INJECTION: 
                f"A {name} vulnerability was detected in an application running on {asset.hostname}. " +
                "This could allow attackers to access or modify database contents. ",
            
            VulnerabilityCategory.XSS: 
                f"A {name} vulnerability was detected in a web application. This could allow attackers " +
                "to execute malicious scripts in users' browsers. ",
            
            VulnerabilityCategory.CSRF: 
                f"A {name} vulnerability was detected. This could allow attackers to trick users " +
                "into performing unwanted actions while authenticated. ",
            
            VulnerabilityCategory.OPEN_PORT: 
                f"An {name} was detected on {asset.ip_address}. " +
                f"Port {random.choice([s['port'] for s in asset.services]) if asset.services else random.randint(1000, 9999)} " +
                "is open but appears to be unnecessary for business operations. ",
            
            VulnerabilityCategory.UNNECESSARY_SERVICE: 
                f"An {name} was detected on {asset.hostname}. " +
                f"The service {random.choice([s['name'] for s in asset.services]) if asset.services else 'unknown'} " +
                "is running but does not appear to be necessary for business operations. ",
            
            VulnerabilityCategory.INFORMATION_DISCLOSURE: 
                f"An {name} vulnerability was detected on {asset.hostname}. The system is revealing " +
                "information that could aid attackers in targeting the system. ",
            
            VulnerabilityCategory.FILE_INCLUSION: 
                f"A {name} vulnerability was detected in an application. This could allow attackers " +
                "to include and execute unauthorized files. "
        }
        
        # Get base description and add some random impacts
        base_desc = descriptions.get(category, f"A {name} vulnerability was detected on {asset.hostname}. ")
        
        impacts = [
            "This may allow attackers to gain unauthorized access to the system.",
            "This vulnerability could lead to data breaches if exploited.",
            "If exploited, this could result in system compromise.",
            "This issue poses a risk to the confidentiality of data stored on the system.",
            "This vulnerability threatens the integrity of the affected system.",
            "Exploitation could lead to privilege escalation.",
            "This vulnerability could be used as part of a larger attack chain.",
            "This issue exposes sensitive system information to potential attackers."
        ]
        
        return base_desc + random.choice(impacts)


# =============================================================================
# SCANNER INTEGRATION SIMULATIONS
# =============================================================================

class ScannerIntegration:
    """Base class for scanner integrations."""
    
    def __init__(self, name: str):
        self.name = name
    
    def scan_asset(self, asset: Asset) -> List[Dict]:
        """Simulate scanning a single asset."""
        raise NotImplementedError("Scanner classes must implement this method")
    
    def scan_assets(self, assets: List[Asset]) -> List[Dict]:
        """Scan multiple assets and aggregate results."""
        all_results = []
        for asset in assets:
            results = self.scan_asset(asset)
            all_results.extend(results)
        return all_results


class NessusScanner(ScannerIntegration):
    """Simulated Nessus scanner integration."""
    
    def __init__(self):
        super().__init__("Nessus")
    
    def scan_asset(self, asset: Asset) -> List[Dict]:
        """Simulate scanning a single asset with Nessus."""
        # Simulate scan duration
        scan_duration = random.uniform(0.1, 0.5)
        time.sleep(scan_duration)
        
        # Generate between 0-4 findings with 70% probability of at least one
        findings_count = random.choices(
            [0, 1, 2, 3, 4], 
            weights=[0.3, 0.3, 0.2, 0.1, 0.1], 
            k=1
        )[0]
        
        results = []
        if findings_count > 0:
            # Select random vulnerability categories, weighted toward certain types
            # Nessus is good at finding misconfigurations, outdated software, and missing patches
            weighted_categories = [
                VulnerabilityCategory.MISCONFIGURATION,
                VulnerabilityCategory.OUTDATED_SOFTWARE,
                VulnerabilityCategory.MISSING_PATCH,
                VulnerabilityCategory.OPEN_PORT,
                VulnerabilityCategory.UNNECESSARY_SERVICE,
                VulnerabilityCategory.DEFAULT_CREDENTIALS
            ]
            
            # Generate findings
            categories = random.choices(weighted_categories, k=findings_count)
            
            for category in categories:
                name = random.choice(SimulatedDataGenerator.VULNERABILITY_NAMES[category])
                cvss = SimulatedDataGenerator._generate_cvss_for_category(category)
                
                # Format in Nessus-like structure
                result = {
                    "plugin_id": random.randint(10000, 99999),
                    "severity": VulnerabilitySeverity.from_cvss(cvss).value,
                    "name": name,
                    "description": SimulatedDataGenerator._generate_description(name, category, asset),
                    "cvss_base_score": cvss,
                    "solution": random.choice(REMEDIATION_DATABASE.get(category, ["Update system"])),
                    "host": {
                        "hostname": asset.hostname,
                        "ip": asset.ip_address
                    },
                    "protocol": random.choice(["tcp", "udp"]),
                    "port": random.choice([s["port"] for s in asset.services]) if asset.services else None,
                    "plugin_output": f"Detected {category.value} on {asset.hostname}",
                    "category": category.value,
                    "scanner": self.name
                }
                results.append(result)
        
        return results


class OpenVASScanner(ScannerIntegration):
    """Simulated OpenVAS scanner integration."""
    
    def __init__(self):
        super().__init__("OpenVAS")
    
    def scan_asset(self, asset: Asset) -> List[Dict]:
        """Simulate scanning a single asset with OpenVAS."""
        # Simulate scan duration
        scan_duration = random.uniform(0.1, 0.5)
        time.sleep(scan_duration)
        
        # Generate findings with 60% probability of at least one
        findings_count = random.choices(
            [0, 1, 2, 3], 
            weights=[0.4, 0.3, 0.2, 0.1], 
            k=1
        )[0]
        
        results = []
        if findings_count > 0:
            # OpenVAS is good at finding these vulnerability types
            weighted_categories = [
                VulnerabilityCategory.SQL_INJECTION,
                VulnerabilityCategory.XSS,
                VulnerabilityCategory.CSRF,
                VulnerabilityCategory.OUTDATED_SOFTWARE,
                VulnerabilityCategory.FILE_INCLUSION,
                VulnerabilityCategory.INFORMATION_DISCLOSURE
            ]
            
            # Generate findings
            categories = random.choices(weighted_categories, k=findings_count)
            
            for category in categories:
                name = random.choice(SimulatedDataGenerator.VULNERABILITY_NAMES[category])
                cvss = SimulatedDataGenerator._generate_cvss_for_category(category)
                
                # Format in OpenVAS-like structure
                result = {
                    "oid": f"1.3.6.1.4.1.25623.1.0.{random.randint(100000, 999999)}",
                    "threat": VulnerabilitySeverity.from_cvss(cvss).name,
                    "name": name,
                    "description": SimulatedDataGenerator._generate_description(name, category, asset),
                    "cvss_base": cvss,
                    "solution_type": "Mitigation",
                    "solution": random.choice(REMEDIATION_DATABASE.get(category, ["Update system"])),
                    "host": {
                        "hostname": asset.hostname,
                        "ip": asset.ip_address
                    },
                    "port": random.choice([s["port"] for s in asset.services]) if asset.services else None,
                    "nvt_category": category.value,
                    "scanner": self.name
                }
                results.append(result)
        
        return results
# =============================================================================
# VULNERABILITY PROCESSING AND ENRICHMENT
# =============================================================================

class VulnerabilityProcessor:
    """Process and normalize scanner results into Vulnerability objects."""
    
    def __init__(self, assets: List[Asset]):
        self.assets = {asset.ip_address: asset for asset in assets}
    
    def process_scanner_results(self, scanner_results: List[Dict]) -> List[Vulnerability]:
        """Process scanner results and convert to normalized Vulnerability objects."""
        vulnerabilities = []
        
        for result in scanner_results:
            # Extract common fields based on scanner type
            if result.get("scanner") == "Nessus":
                vulnerability = self._process_nessus_result(result)
            elif result.get("scanner") == "OpenVAS":
                vulnerability = self._process_openvas_result(result)
            elif result.get("scanner") == "QualysGuard":
                vulnerability = self._process_qualys_result(result)
            else:
                continue  # Skip unknown scanner types
            
            if vulnerability:
                vulnerabilities.append(vulnerability)
        
        return vulnerabilities
    
    def _process_nessus_result(self, result: Dict) -> Optional[Vulnerability]:
        """Process a Nessus scan result."""
        # Get the affected asset
        ip_address = result.get("host", {}).get("ip")
        if not ip_address or ip_address not in self.assets:
            return None
        
        asset = self.assets[ip_address]
        
        # Determine vulnerability category
        category_str = result.get("category")
        try:
            category = next(c for c in VulnerabilityCategory if c.value == category_str)
        except StopIteration:
            # Default to misconfiguration if category doesn't match
            category = VulnerabilityCategory.MISCONFIGURATION
        
        # Create the vulnerability object
        vulnerability = Vulnerability(
            vuln_id=str(uuid.uuid4()),
            name=result.get("name", "Unknown Vulnerability"),
            description=result.get("description", "No description available"),
            cvss_score=result.get("cvss_base_score", 5.0),
            category=category,
            affected_asset=asset,
            detection_date=datetime.datetime.now(),
            remediation_steps=[result.get("solution", "No solution provided")],
            scanner_source=result.get("scanner")
        )
        
        return vulnerability
    
    def _process_openvas_result(self, result: Dict) -> Optional[Vulnerability]:
        """Process an OpenVAS scan result."""
        # Get the affected asset
        ip_address = result.get("host", {}).get("ip")
        if not ip_address or ip_address not in self.assets:
            return None
        
        asset = self.assets[ip_address]
        
        # Determine vulnerability category
        category_str = result.get("nvt_category")
        try:
            category = next(c for c in VulnerabilityCategory if c.value == category_str)
        except StopIteration:
            # Default to information disclosure if category doesn't match
            category = VulnerabilityCategory.INFORMATION_DISCLOSURE
        
        # Create the vulnerability object
        vulnerability = Vulnerability(
            vuln_id=str(uuid.uuid4()),
            name=result.get("name", "Unknown Vulnerability"),
            description=result.get("description", "No description available"),
            cvss_score=result.get("cvss_base", 5.0),
            category=category,
            affected_asset=asset,
            detection_date=datetime.datetime.now(),
            remediation_steps=[result.get("solution", "No solution provided")],
            scanner_source=result.get("scanner")
        )
        
        return vulnerability
    
    def _process_qualys_result(self, result: Dict) -> Optional[Vulnerability]:
        """Process a Qualys scan result."""
        # Get the affected asset
        ip_address = result.get("asset", {}).get("ip")
        if not ip_address or ip_address not in self.assets:
            return None
        
        asset = self.assets[ip_address]
        
        # Determine vulnerability category
        category_str = result.get("category")
        try:
            category = next(c for c in VulnerabilityCategory if c.value == category_str)
        except StopIteration:
            # Default to outdated software if category doesn't match
            category = VulnerabilityCategory.OUTDATED_SOFTWARE
        
        # Create the vulnerability object
        detection_date = datetime.datetime.fromisoformat(result.get("first_found", datetime.datetime.now().isoformat()))
        
        vulnerability = Vulnerability(
            vuln_id=str(uuid.uuid4()),
            name=result.get("title", "Unknown Vulnerability"),
            description=result.get("description", "No description available"),
            cvss_score=result.get("cvss_base", 5.0),
            category=category,
            affected_asset=asset,
            detection_date=detection_date,
            remediation_steps=[result.get("solution", "No solution provided")],
            scanner_source=result.get("scanner")
        )
        
        return vulnerability


class VulnerabilityEnricher:
    """Enrich vulnerability data with additional information."""
    
    def __init__(self):
        # Simulated threat intelligence database
        self.threat_intel = self._initialize_threat_intel()
    
    def _initialize_threat_intel(self) -> Dict[str, Dict]:
        """Initialize simulated threat intelligence data."""
        threat_intel = {}
        
        # Create some random CVEs and associate with threat intel
        for i in range(20):
            year = random.randint(2018, 2024)
            cve_id = f"CVE-{year}-{random.randint(1000, 9999)}"
            
            # Simulated threat actor information
            threat_actor = random.choice([
                "APT28", "Lazarus Group", "Sandworm", "Fancy Bear", 
                "Equation Group", "Dark Halo", "Cobalt Group", "FIN7"
            ])
            
            # Simulated campaign information
            campaign = random.choice([
                "Operation Ghost", "Cloud Hopper", "Olympic Games", 
                "NotPetya", "Solar Wind", "Colonial Pipeline", "Sunburst"
            ])
            
            # Simulated exploit information
            exploit_available = random.random() < 0.7
            exploit_in_wild = exploit_available and random.random() < 0.6
            exploit_price = random.randint(5000, 50000) if exploit_available else None
            
            # Simulated affected product information
            affected_products = random.sample([
                "Windows 10", "Ubuntu 20.04", "Apache 2.4", "Nginx 1.18", 
                "MySQL 8.0", "PostgreSQL 13", "Redis 6.0", "WordPress 5.7",
                "Exchange Server 2019", "Oracle Database 19c"
            ], k=random.randint(1, 3))
            
            # Create the threat intel entry
            threat_intel[cve_id] = {
                "threat_actors": [threat_actor],
                "campaigns": [campaign] if random.random() < 0.5 else [],
                "exploit_available": exploit_available,
                "exploit_in_wild": exploit_in_wild,
                "exploit_price": exploit_price,
                "mitigation_complexity": random.choice(["Low", "Medium", "High"]),
                "affected_products": affected_products,
                "estimated_remediation_time": f"{random.randint(1, 8)} hours",
                "last_observed": datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 365))
            }
        
        return threat_intel
    
    def enrich_vulnerability(self, vulnerability: Vulnerability) -> Vulnerability:
        """Enrich a vulnerability with additional information."""
        # Enrich with threat intelligence if CVE is available
        if vulnerability.cve_id and vulnerability.cve_id in self.threat_intel:
            intel = self.threat_intel[vulnerability.cve_id]
            
            # Adjust exploit status based on threat intel
            if intel["exploit_in_wild"]:
                vulnerability.exploit_status = ExploitStatus.PUBLIC
            elif intel["exploit_available"]:
                vulnerability.exploit_status = ExploitStatus.POC
        
        # Simulate enrichment with vulnerability database information
        # Here we can enhance the description, add more detailed remediation steps, etc.
        if random.random() < 0.3:  # 30% chance to enhance remediation
            category = vulnerability.category
            if category in REMEDIATION_DATABASE:
                # Add more specific remediation steps
                vulnerability.remediation_steps = REMEDIATION_DATABASE[category]
        
        return vulnerability


class ExploitValidator:
    """Simulate validation of vulnerability exploitability using Metasploit-like functionality."""
    
    def validate_exploitability(self, vulnerability: Vulnerability) -> str:
        """Simulate validation of whether a vulnerability is exploitable."""
        # Higher CVSS scores are more likely to be exploitable
        exploit_chance = min(0.9, vulnerability.cvss_score / 10 * 0.8)
        
        # Some vulnerability categories are more likely to have working exploits
        category_multipliers = {
            VulnerabilityCategory.SQL_INJECTION: 1.5,
            VulnerabilityCategory.XSS: 1.3,
            VulnerabilityCategory.FILE_INCLUSION: 1.4,
            VulnerabilityCategory.DEFAULT_CREDENTIALS: 1.8,
            VulnerabilityCategory.WEAK_AUTHENTICATION: 1.5,
            VulnerabilityCategory.MISSING_PATCH: 1.3,
            VulnerabilityCategory.OUTDATED_SOFTWARE: 1.2,
        }
        
        multiplier = category_multipliers.get(vulnerability.category, 1.0)
        exploit_chance *= multiplier
        
        # Exploit status also affects validation
        status_multipliers = {
            ExploitStatus.PUBLIC: 1.8,
            ExploitStatus.POC: 1.5,
            ExploitStatus.PRIVATE: 1.2,
            ExploitStatus.THEORETICAL: 0.5,
            ExploitStatus.NONE: 0.1,
        }
        
        if vulnerability.exploit_status:
            multiplier = status_multipliers.get(vulnerability.exploit_status, 1.0)
            exploit_chance *= multiplier
        
        # Cap at 95% chance
        exploit_chance = min(0.95, exploit_chance)
        
        # Determine if exploitable
        exploitable = random.random() < exploit_chance
        
        # Simulate Metasploit/validation output
        if exploitable:
            result = random.choice([
                "Exploit successful - Shell access obtained",
                "Exploit successful - Authentication bypass achieved",
                "Exploit successful - Data extraction possible",
                "Exploit successful - Privilege escalation confirmed",
                "Exploit successful - Remote code execution verified"
            ])
            vulnerability.validation_status = "Confirmed Exploitable"
        else:
            result = random.choice([
                "Exploit failed - Target not vulnerable",
                "Exploit failed - System patched",
                "Exploit failed - Insufficient permissions",
                "Exploit failed - Protection mechanisms in place",
                "Exploit failed - Target configuration not vulnerable"
            ])
            vulnerability.validation_status = "Not Exploitable"
        
        # Simulate a delay in validation
        time.sleep(random.uniform(0.1, 0.3))
        
        return result


# =============================================================================
# VULNERABILITY SCORING AND PRIORITIZATION
# =============================================================================

class VulnerabilityScoringEngine:
    """Score and prioritize vulnerabilities based on multiple factors."""
    
    def calculate_risk_score(self, vulnerability: Vulnerability) -> float:
        """Calculate a risk score from 0-100 for the vulnerability."""
        # Base score from CVSS (0-10 scale converted to 0-50)
        base_score = vulnerability.cvss_score * 5
        
        # Asset criticality factor (0-25)
        criticality_factor = vulnerability.affected_asset.criticality.value * 6.25
        
        # Exploit availability factor (0-15)
        exploit_weights = {
            ExploitStatus.PUBLIC: 15,
            ExploitStatus.PRIVATE: 12,
            ExploitStatus.POC: 9,
            ExploitStatus.THEORETICAL: 5,
            ExploitStatus.NONE: 0
        }
        exploit_factor = exploit_weights.get(vulnerability.exploit_status, 0)
        
        # Time factor - newer vulnerabilities get higher priority (0-10)
        days_since_detection = (datetime.datetime.now() - vulnerability.detection_date).days
        time_factor = max(0, 10 - (days_since_detection / 10))
        
        # Calculate final score
        risk_score = base_score + criticality_factor + exploit_factor + time_factor
        
        # Cap at 100
        risk_score = min(100, risk_score)
        
        # Store the score in the vulnerability object
        vulnerability.risk_score = round(risk_score, 1)
        
        return vulnerability.risk_score
    
    def prioritize_vulnerabilities(self, vulnerabilities: List[Vulnerability]) -> List[Vulnerability]:
        """Score and sort vulnerabilities by risk score."""
        # Calculate risk scores for all vulnerabilities
        for vuln in vulnerabilities:
            self.calculate_risk_score(vuln)
        
        # Sort by risk score (high to low)
        return sorted(vulnerabilities, key=lambda v: v.risk_score, reverse=True)


# =============================================================================
# CONTINUOUS MONITORING AND REPORTING
# =============================================================================

class VulnerabilityReportGenerator:
    """Generate reports from vulnerability data."""
    
    def generate_summary_report(self, vulnerabilities: List[Vulnerability], assets: List[Asset]) -> Dict:
        """Generate a summary report of vulnerability findings."""
        # Count vulnerabilities by severity
        severity_counts = {severity.name: 0 for severity in VulnerabilityCategory}
        for vuln in vulnerabilities:
            severity_counts[vuln.severity.name] += 1
        
        # Count vulnerabilities by category
        category_counts = {category.name: 0 for category in VulnerabilityCategory}
        for vuln in vulnerabilities:
            category_counts[vuln.category.name] += 1
        
        # Count vulnerabilities by asset criticality
        criticality_counts = {criticality.name: 0 for criticality in AssetCriticality}
        for vuln in vulnerabilities:
            criticality_counts[vuln.affected_asset.criticality.name] += 1
        
        # Calculate average risk score
        avg_risk_score = sum(v.risk_score or 0 for v in vulnerabilities) / len(vulnerabilities) if vulnerabilities else 0
        
        # Find top 10 highest risk vulnerabilities
        top_vulns = sorted(vulnerabilities, key=lambda v: v.risk_score or 0, reverse=True)[:10]
        top_vulns_data = [
            {
                "name": v.name,
                "cvss": v.cvss_score,
                "risk_score": v.risk_score,
                "asset": v.affected_asset.hostname,
                "severity": v.severity.name
            }
            for v in top_vulns
        ]
        
        # Find most vulnerable assets (assets with most high/critical vulns)
        asset_vuln_counts = {}
        for asset in assets:
            high_critical_count = sum(1 for v in asset.vulnerabilities 
                                     if v.severity in [VulnerabilitySeverity.HIGH, VulnerabilitySeverity.CRITICAL])
            asset_vuln_counts[asset.hostname] = {
                "asset_id": asset.asset_id,
                "hostname": asset.hostname,
                "ip": asset.ip_address,
                "os": asset.os,
                "criticality": asset.criticality.name,
                "high_critical_count": high_critical_count,
                "total_vulnerabilities": len(asset.vulnerabilities)
            }
        
        most_vulnerable_assets = sorted(
            asset_vuln_counts.values(), 
            key=lambda a: (a["high_critical_count"], a["total_vulnerabilities"]), 
            reverse=True
        )[:5]
        
        # Build the report
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": {
                "total_assets_scanned": len(assets),
                "total_vulnerabilities": len(vulnerabilities),
                "average_risk_score": round(avg_risk_score, 1),
                "vulnerability_counts_by_severity": severity_counts,
                "vulnerability_counts_by_category": category_counts,
                "vulnerability_counts_by_asset_criticality": criticality_counts,
            },
            "top_vulnerabilities": top_vulns_data,
            "most_vulnerable_assets": most_vulnerable_assets,
            "recent_changes": {
                "new_vulnerabilities": len([v for v in vulnerabilities 
                                          if (datetime.datetime.now() - v.detection_date).days < 7]),
                "remediated_vulnerabilities": random.randint(0, 10),  # Simulated remediation count
            }
        }
        
        return report


class ContinuousMonitor:
    """Simulate continuous monitoring of assets and vulnerabilities."""
    
    def __init__(
        self, 
        assets: List[Asset], 
        scanners: List[ScannerIntegration],
        processor: VulnerabilityProcessor,
        enricher: VulnerabilityEnricher,
        scoring_engine: VulnerabilityScoringEngine,
        report_generator: VulnerabilityReportGenerator,
        schedule_minutes: int = 60
    ):
        self.assets = assets
        self.scanners = scanners
        self.processor = processor
        self.enricher = enricher
        self.scoring_engine = scoring_engine
        self.report_generator = report_generator
        self.schedule_minutes = schedule_minutes
        self.vulnerabilities = []
        self.scan_history = []
    
    def simulate_scheduled_scan(self) -> Dict:
        """Simulate a scheduled scan and return results."""
        scan_start_time = datetime.datetime.now()
        
        # Scan with each scanner
        all_results = []
        for scanner in self.scanners:
            results = scanner.scan_assets(self.assets)
            all_results.extend(results)
        
        # Process results into vulnerabilities
        new_vulnerabilities = self.processor.process_scanner_results(all_results)
        
        # Enrich vulnerabilities
        for vuln in new_vulnerabilities:
            self.enricher.enrich_vulnerability(vuln)
        
        # Score vulnerabilities
        for vuln in new_vulnerabilities:
            self.scoring_engine.calculate_risk_score(vuln)
        
        # Update vulnerability list
        self.vulnerabilities.extend(new_vulnerabilities)
        
        # Generate report
        scan_report = {
            "scan_start_time": scan_start_time.isoformat(),
            "scan_end_time": datetime.datetime.now().isoformat(),
            "scanner_count": len(self.scanners),
            "asset_count": len(self.assets),
            "new_vulnerabilities_count": len(new_vulnerabilities),
            "total_vulnerabilities_count": len(self.vulnerabilities),
            "summary_report": self.report_generator.generate_summary_report(self.vulnerabilities, self.assets)
        }
        
        # Record scan history
        self.scan_history.append(scan_report)
        
        return scan_report


# =============================================================================
# MACHINE LEARNING FOR ANOMALY DETECTION
# =============================================================================

class AnomalyDetector:
    """Use machine learning to detect anomalies in vulnerability scan results."""
    
    def __init__(self):
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.is_trained = False
    
    def prepare_data(self, vulnerabilities: List[Vulnerability]) -> pd.DataFrame:
        """Prepare vulnerability data for anomaly detection."""
        # Extract features from vulnerabilities
        data = []
        for vuln in vulnerabilities:
            features = {
                "cvss_score": vuln.cvss_score,
                "risk_score": vuln.risk_score or 0,
                "asset_criticality": vuln.affected_asset.criticality.value,
                "severity": vuln.severity.value,
                "category": hash(vuln.category.name) % 100,  # Simple hash to convert category to numeric
                "has_exploit": 1 if vuln.exploit_status in [ExploitStatus.PUBLIC, ExploitStatus.POC] else 0,
                "exploit_status": hash(vuln.exploit_status.name) % 100 if vuln.exploit_status else 0,
                "scanner_source_hash": hash(vuln.scanner_source) % 100 if vuln.scanner_source else 0,
            }
            data.append(features)
        
        return pd.DataFrame(data)
    
    def train(self, vulnerabilities: List[Vulnerability]) -> None:
        """Train the anomaly detection model."""
        if len(vulnerabilities) < 10:
            print("Warning: Not enough vulnerabilities to train model")
            return
        
        data = self.prepare_data(vulnerabilities)
        self.model.fit(data)
        self.is_trained = True
    
    def detect_anomalies(self, vulnerabilities: List[Vulnerability]) -> List[Vulnerability]:
        """Detect anomalies in vulnerabilities."""
        if not self.is_trained:
            print("Warning: Model not trained yet")
            return []
        
        data = self.prepare_data(vulnerabilities)
        predictions = self.model.predict(data)
        
        # Isolation Forest returns -1 for anomalies and 1 for normal data
        anomalies = [vuln for i, vuln in enumerate(vulnerabilities) if predictions[i] == -1]
        return anomalies
    
    def get_anomaly_scores(self, vulnerabilities: List[Vulnerability]) -> Dict[str, float]:
        """Get anomaly scores for vulnerabilities."""
        if not self.is_trained:
            print("Warning: Model not trained yet")
            return {}
        
        data = self.prepare_data(vulnerabilities)
        scores = self.model.decision_function(data)
        
        # Convert to a more intuitive scale where higher values indicate more anomalous
        scores = -scores
        
        # Normalize to 0-100 scale
        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score
        if range_score > 0:
            normalized_scores = [(s - min_score) / range_score * 100 for s in scores]
        else:
            normalized_scores = [50 for _ in scores]  # Default to middle if no range
        
        # Map scores to vulnerability IDs
        return {vuln.vuln_id: round(score, 2) for vuln, score in zip(vulnerabilities, normalized_scores)}


# =============================================================================
# STREAMLIT UI
# =============================================================================

def run_streamlit_app():
    """Main function for Streamlit app."""
    import streamlit as st
    
    st.set_page_config(
        page_title="Automated Vulnerability Assessment Tool",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🛡️ Automated Vulnerability Assessment Tool")
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.assets = []
        st.session_state.vulnerabilities = []
        st.session_state.scan_history = []
        st.session_state.anomalies = []
        st.session_state.anomaly_scores = {}
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        if not st.session_state.initialized:
            assets_count = st.slider("Number of assets to simulate", 5, 50, 20)
            vulns_range = st.slider(
                "Vulnerabilities per asset range",
                0, 10, (1, 5),
                help="Minimum and maximum number of vulnerabilities per asset"
            )
            
            if st.button("Initialize System", type="primary"):
                with st.spinner("Initializing system components..."):
                    # Generate assets
                    st.session_state.assets = SimulatedDataGenerator.generate_assets(assets_count)
                    
                    # Generate initial vulnerabilities
                    st.session_state.vulnerabilities = SimulatedDataGenerator.generate_vulnerabilities(
                        st.session_state.assets, 
                        count_range=vulns_range
                    )
                    
                    # Create scanners
                    st.session_state.scanners = [
                        NessusScanner(),
                        OpenVASScanner(),
                    ]
                    
                    # Create other components
                    st.session_state.processor = VulnerabilityProcessor(st.session_state.assets)
                    st.session_state.enricher = VulnerabilityEnricher()
                    st.session_state.scoring_engine = VulnerabilityScoringEngine()
                    st.session_state.report_generator = VulnerabilityReportGenerator()
                    st.session_state.validator = ExploitValidator()
                    st.session_state.anomaly_detector = AnomalyDetector()
                    
                    # Score vulnerabilities
                    for vuln in st.session_state.vulnerabilities:
                        st.session_state.scoring_engine.calculate_risk_score(vuln)
                    
                    # Initialize monitor
                    st.session_state.monitor = ContinuousMonitor(
                        st.session_state.assets,
                        st.session_state.scanners,
                        st.session_state.processor,
                        st.session_state.enricher,
                        st.session_state.scoring_engine,
                        st.session_state.report_generator
                    )
                    
                    # Train anomaly detector
                    if len(st.session_state.vulnerabilities) >= 10:
                        st.session_state.anomaly_detector.train(st.session_state.vulnerabilities)
                        # Detect initial anomalies
                        st.session_state.anomalies = st.session_state.anomaly_detector.detect_anomalies(
                            st.session_state.vulnerabilities
                        )
                        st.session_state.anomaly_scores = st.session_state.anomaly_detector.get_anomaly_scores(
                            st.session_state.vulnerabilities
                        )
                    
                    st.session_state.initialized = True
                    st.success("System initialized!")
        else:
            if st.button("Run Vulnerability Scan", type="primary"):
                with st.spinner("Running vulnerability scan..."):
                    scan_result = st.session_state.monitor.simulate_scheduled_scan()
                    st.session_state.vulnerabilities = st.session_state.monitor.vulnerabilities
                    st.session_state.scan_history = st.session_state.monitor.scan_history
                    
                    # Retrain anomaly detector with new data
                    if len(st.session_state.vulnerabilities) >= 10:
                        st.session_state.anomaly_detector.train(st.session_state.vulnerabilities)
                        st.session_state.anomalies = st.session_state.anomaly_detector.detect_anomalies(
                            st.session_state.vulnerabilities
                        )
                        st.session_state.anomaly_scores = st.session_state.anomaly_detector.get_anomaly_scores(
                            st.session_state.vulnerabilities
                        )
                    
                    st.success(f"Scan complete! Found {scan_result['new_vulnerabilities_count']} new vulnerabilities.")
            
            if st.button("Reset System"):
                st.session_state.initialized = False
                st.session_state.assets = []
                st.session_state.vulnerabilities = []
                st.session_state.scan_history = []
                st.session_state.anomalies = []
                st.session_state.anomaly_scores = {}
                st.experimental_rerun()
    
    # Main content
    if st.session_state.initialized:
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Dashboard", 
            "Vulnerabilities", 
            "Assets", 
            "Anomaly Detection",
            "Reports"
        ])
        
    with tab1:
            # Dashboard view
            st.header("Vulnerability Dashboard")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_vulns = len(st.session_state.vulnerabilities)
            high_critical_vulns = sum(1 for v in st.session_state.vulnerabilities 
                                    if v.severity in [VulnerabilitySeverity.HIGH, VulnerabilitySeverity.CRITICAL])
            
            with col1:
                st.metric("Total Assets", len(st.session_state.assets))
            
            with col2:
                st.metric("Total Vulnerabilities", total_vulns)
            
            with col3:
                st.metric("High/Critical Vulnerabilities", high_critical_vulns)
            
            with col4:
                if st.session_state.anomalies:
                    st.metric("Anomalous Findings", len(st.session_state.anomalies))
                else:
                    st.metric("Anomalous Findings", "N/A")
            
            # Charts for visual representation
            st.subheader("Vulnerabilities by Severity")
            severity_counts = {severity.name: 0 for severity in VulnerabilitySeverity}
            for vuln in st.session_state.vulnerabilities:
                severity_counts[vuln.severity.name] += 1
            severity_df = pd.DataFrame(list(severity_counts.items()), columns=["Severity", "Count"])
            st.bar_chart(severity_df, x="Severity", y="Count")
                    
            st.subheader("Vulnerabilities by Category")
            category_counts = {category.name: 0 for category in VulnerabilityCategory}
            for vuln in st.session_state.vulnerabilities:
                category_counts[vuln.category.name] += 1
            category_df = pd.DataFrame(list(category_counts.items()), columns=["Category", "Count"])
            st.bar_chart(category_df, x="Category", y="Count")

    with tab2:
        # Vulnerability list view
        st.header("Vulnerability List")
        
        if st.session_state.vulnerabilities:
            # Create a DataFrame for display
            vuln_data = [
                {
                    "ID": vuln.vuln_id,
                    "Name": vuln.name,
                    "Severity": vuln.severity.name,
                    "CVSS": vuln.cvss_score,
                    "Risk Score": vuln.risk_score,
                    "Category": vuln.category.name,
                    "Asset": vuln.affected_asset.hostname,
                    "Exploit Status": vuln.exploit_status.name if vuln.exploit_status else "Unknown",
                    "Validation Status": vuln.validation_status,
                    "Detection Date": vuln.detection_date.strftime("%Y-%m-%d"),
                    "Anomaly Score": st.session_state.anomaly_scores.get(vuln.vuln_id, "N/A")
                }
                for vuln in st.session_state.vulnerabilities
            ]
            
            vuln_df = pd.DataFrame(vuln_data)
            
            # Allow sorting by columns
            sort_column = st.selectbox("Sort by", vuln_df.columns)
            ascending = st.checkbox("Ascending order")
            vuln_df_sorted = vuln_df.sort_values(by=sort_column, ascending=ascending)
            
            # Display DataFrame
            st.dataframe(vuln_df_sorted, use_container_width=True)
            
            # Vulnerability details
            selected_vuln_id = st.selectbox("Select vulnerability for details", [v.vuln_id for v in st.session_state.vulnerabilities])
            
            selected_vuln = next((v for v in st.session_state.vulnerabilities if v.vuln_id == selected_vuln_id), None)
            
            if selected_vuln:
                st.subheader(f"Vulnerability Details: {selected_vuln.name}")
                st.write(f"Description: {selected_vuln.description}")
                st.write(f"CVSS Score: {selected_vuln.cvss_score}")
                st.write(f"Risk Score: {selected_vuln.risk_score}")
                st.write(f"Category: {selected_vuln.category.name}")
                st.write(f"Affected Asset: {selected_vuln.affected_asset.hostname}")
                st.write(f"Exploit Status: {selected_vuln.exploit_status.name if selected_vuln.exploit_status else 'Unknown'}")
                st.write(f"Validation Status: {selected_vuln.validation_status}")
                st.write(f"Detection Date: {selected_vuln.detection_date.strftime('%Y-%m-%d')}")
                st.write(f"Scanner Source: {selected_vuln.scanner_source}")
                
                # Remediation steps
                st.subheader("Remediation Steps")
                if selected_vuln.remediation_steps:
                    for i, step in enumerate(selected_vuln.remediation_steps):
                        st.write(f"{i+1}. {step}")
                else:
                    st.write("No remediation steps available for this vulnerability.")
                
                # Exploit validation
                if st.button("Validate Exploit", key=selected_vuln.vuln_id):
                    with st.spinner("Validating exploit..."):
                        result = st.session_state.validator.validate_exploitability(selected_vuln)
                        st.write(result)
                        # Update validation status in session state
                        st.session_state.vulnerabilities = [
                            v if v.vuln_id != selected_vuln.vuln_id else selected_vuln
                            for v in st.session_state.vulnerabilities
                        ]
                        st.experimental_rerun()
        else:
            st.info("No vulnerabilities found yet. Run a scan to get started.")
    with tab3:
        # Asset list view
        st.header("Asset List")
        
        if st.session_state.assets:
            asset_data = [
                {
                    "ID": asset.asset_id,
                    "Hostname": asset.hostname,
                    "IP Address": asset.ip_address,
                    "OS": asset.os,
                    "Criticality": asset.criticality.name,
                    "Department": asset.department,
                    "Vulnerabilities": len(asset.vulnerabilities),
                }
                for asset in st.session_state.assets
            ]
            
            asset_df = pd.DataFrame(asset_data)
            
            # Allow sorting by columns
            sort_column = st.selectbox("Sort by", asset_df.columns)
            ascending = st.checkbox("Ascending order")
            asset_df_sorted = asset_df.sort_values(by=sort_column, ascending=ascending)
            
            # Display DataFrame
            st.dataframe(asset_df_sorted, use_container_width=True)
            
            # Asset details
            selected_asset_id = st.selectbox("Select asset for details", [a.asset_id for a in st.session_state.assets])
            
            selected_asset = next((a for a in st.session_state.assets if a.asset_id == selected_asset_id), None)
            
            if selected_asset:
                st.subheader(f"Asset Details: {selected_asset.hostname}")
                st.write(f"IP Address: {selected_asset.ip_address}")
                st.write(f"OS: {selected_asset.os}")
                st.write(f"Criticality: {selected_asset.criticality.name}")
                st.write(f"Department: {selected_asset.department}")
                
                # Display services
                if selected_asset.services:
                    st.subheader("Services")
                    for service in selected_asset.services:
                        st.write(f"Name: {service['name']}, Port: {service['port']}, Version: {service['version']}")
                else:
                    st.write("No services found on this asset.")
                
                # Display vulnerabilities
                st.subheader("Vulnerabilities")
                if selected_asset.vulnerabilities:
                    for vuln in selected_asset.vulnerabilities:
                        st.write(
                            f"{vuln.name} - Severity: {vuln.severity.name}, CVSS: {vuln.cvss_score}, Risk Score: {vuln.risk_score}"
                        )
                else:
                    st.write("No vulnerabilities found on this asset.")
        else:
            st.info("No assets found yet. Initialize the system to get started.")

    with tab4:
        # Anomaly detection view
        st.header("Anomaly Detection")
        
        if st.session_state.anomalies:
            st.subheader("Anomalous Vulnerabilities")
            
            # Display anomalies in a table
            anomaly_data = [
                {
                    "ID": vuln.vuln_id,
                    "Name": vuln.name,
                    "Severity": vuln.severity.name,
                    "CVSS": vuln.cvss_score,
                    "Risk Score": vuln.risk_score,
                    "Category": vuln.category.name,
                    "Asset": vuln.affected_asset.hostname,
                    "Anomaly Score": st.session_state.anomaly_scores.get(vuln.vuln_id)
                }
                for vuln in st.session_state.anomalies
            ]
            
            anomaly_df = pd.DataFrame(anomaly_data)
            st.dataframe(anomaly_df, use_container_width=True)
            
            # Explain what makes these vulnerabilities anomalous
            st.write(
                """
                These vulnerabilities were flagged as anomalous because they deviate
                significantly from the typical patterns observed in the vulnerability data.
                This could indicate:
                
                * New or unusual vulnerability types: The detected vulnerabilities might belong to
                    categories or have characteristics that haven't been seen before.
                * Unexpected changes in vulnerability patterns: There might be a sudden increase
                    in the severity or risk of vulnerabilities on certain assets.
                * Suspicious activity: The anomalies could be indicative of malicious activity
                    or targeted attacks.
                
                It is important to investigate these anomalous vulnerabilities further to
                determine the root cause and take appropriate action.
                """
            )
        else:
            st.info("No anomalies detected yet.")

    with tab5:
        # Reports tab
        st.header("Vulnerability Reports")
        
        # Display latest scan report
        if st.session_state.scan_history:
            latest_report = st.session_state.scan_history[-1]["summary_report"]
            
            st.subheader("Latest Scan Summary")
            st.write(latest_report["summary"])
