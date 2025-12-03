import React, { useEffect } from "react";
import { Container, Row, Col, Card, Button, Image } from "react-bootstrap";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../modules/auth";
import heroImage from "../assets/ai.png";

function HomePage() {
    const navigate = useNavigate();
    const { user, loading } = useAuth();

    // Redirect logged-in users to dashboard
    useEffect(() => {
        if (!loading && user) {
            navigate("/Dashboard", { replace: true });
        }
    }, [user, loading, navigate]);

    // Show loading state while checking auth
    if (loading) {
        return (
            <Container className="mt-5 text-center">
                <div>Loading...</div>
            </Container>
        );
    }

    // Don't render homepage if user is logged in (will redirect)
    if (user) {
        return null;
    }

    return (
        <Container className="mt-5">
            <Row className="justify-content-center mb-4">
                <Col md={8} className="text-center">
                    <h1 className="display-4 p-2">LET: LLM Explanation Tool</h1>
                    <Image src={heroImage} fluid rounded className="mb-4 shadow"/>

                    <p className="lead">
                    "Explore the faithfulness and plausibility of AI explanations with LET. Upload datasets or test single entries, run models from any major provider or BERT locally, and compare self-explanations, post-hoc rationales, and SHAP-augmented insights. Designed for research and user studies, LET makes explanation quality measurable and transparent."
                    </p>
                </Col>
            </Row>
            <Row className="justify-content-center">
                <Col md={5}>
                    <Card className="shadow mb-4">
                        <Card.Body className="text-center">
                            <Card.Title>
                                <i className="bi bi-box-arrow-in-right me-2"></i>
                                Login
                            </Card.Title>
                            <Card.Text>
                                Sign in to access your account and start using LET.
                            </Card.Text>
                            <Button variant="dark" onClick={() => navigate("/login")} className="w-100">
                                Go to Login
                            </Button>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={5}>
                    <Card className="shadow mb-4">
                        <Card.Body className="text-center">
                            <Card.Title>
                                <i className="bi bi-person-plus me-2"></i>
                                Register
                            </Card.Title>
                            <Card.Text>
                                Create a new account to get started with LET.
                            </Card.Text>
                            <Button variant="dark" onClick={() => navigate("/register")} className="w-100">
                                Go to Register
                            </Button>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
        </Container>
    );
}

export default HomePage;