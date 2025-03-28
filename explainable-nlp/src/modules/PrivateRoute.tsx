import { Navigate } from "react-router-dom";
import {useAuth} from "./auth";

interface PrivateRouteProps {
    element: React.ReactElement;
}


const PrivateRoute = ({ element }: PrivateRouteProps) => {
    const { user, loading } = useAuth();

    if (loading) {
        return <div>Loading...</div>; // Or a spinner component
    }

    return user ? element : <Navigate to="/login" replace />;
};

export default PrivateRoute;