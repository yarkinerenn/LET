import { Button } from 'react-bootstrap';
import {useAuth} from "../modules/auth";
import App from "../App";

const Dashboard = () => {
    const { user, logout } = useAuth();

    return (
        <div className="py-5 text-center">
            <div className="hero-section mb-5">

                {user ? (  
                    <h1 className="display-4 mb-3">
                        Welcome {user?.username || 'to Auth App'}
                    </h1>
                ) : (
                    <div className="mt-4">
                        <a href="/login" className="btn btn-primary mx-2">
                            Login
                        </a>
                        <a href="/register" className="btn btn-outline-primary mx-2">
                            Register
                        </a>
                    </div>
                )}
            </div>
        </div>
    );
};
export default Dashboard;