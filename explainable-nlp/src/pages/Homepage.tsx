import React from "react";
import { Container, Row, Col, Card, Button, Image } from "react-bootstrap";
import { useNavigate } from "react-router-dom";
import heroImage from "../assets/ai.png"; // Resmi buraya koy

function HomePage() {
    const navigate = useNavigate();

    return (
        <Container className="mt-5">
            <Row className="justify-content-center mb-4">
                <Col md={8} className="text-center">
                    <h1 className="display-4 p-2">XNLP: Explainable NLP Platform</h1>
                    <Image src={heroImage} fluid rounded className="mb-4 shadow"/>

                    <p className="lead">
                        Discover the power of explainable natural language processing!
                        Classify your datasets or individual entries using LLMs or BERT,
                        and gain deep insights with explainability through LLM or SHAP.
                        Utilize few-shot learning and chain of prompting in both explanation and classification.
                    </p>
                </Col>
            </Row>
            <Row className="justify-content-center">
                <Col md={5}>
                    <Card className="shadow mb-4">
                        <Card.Body className="text-center">
                            <Card.Title>Dashboard</Card.Title>
                            <Card.Text>
                                Dive into sentiment classification and real-time analysis.
                            </Card.Text>
                            <Button variant="dark" onClick={() => navigate("/dashboard")}>
                                Go to Dashboard
                            </Button>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={5}>
                    <Card className="shadow mb-4">
                        <Card.Body className="text-center">
                            <Card.Title>Datasets</Card.Title>
                            <Card.Text>
                                Manage and classify your datasets with ease.
                            </Card.Text>
                            <Button variant="dark" onClick={() => navigate("/datasets")}>
                                Go to Datasets
                            </Button>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
        </Container>
    );
}

export default HomePage;