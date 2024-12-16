import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import gennorm
from scipy.stats import norm
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def video_to_3d_array(video_path):
    """
    Convert a video to a 3D numpy array.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        numpy.ndarray: 3D array (frames, height, width).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()
    return np.array(frames)


def apply_3d_wavelet_transform(video_array,wavelet='haar'):
    """
    Apply a 3D Wavelet Transform to a 3D array.

    Parameters:
        video_array (numpy.ndarray): 3D video array.

    Returns:
        tuple: Approximation and detail coefficients.
    """

    coeffs = pywt.dwtn(video_array, wavelet, axes=(0, 1, 2))
    return coeffs['aaa'], coeffs  # Approximation and all components


def plot_residual_slices(details, plane='xz'):
    """
    Plot residual components of the 3D wavelet transform.

    Parameters:
        details (dict): Detail coefficients from the wavelet transform.
        plane (str): The plane to visualize ('xz' or 'yz').
    """
    # Extract details at first level
    residual = details['daa']  # Example: Choose specific detail coefficient

    num_frames, height, width = residual.shape

    if plane == 'xz':
        slice_ = residual[:, :, width // 2]  # Fixed y-axis
        xlabel, ylabel = 'X', 'Z'
    elif plane == 'yz':
        slice_ = residual[:, height // 2, :]  # Fixed x-axis
        xlabel, ylabel = 'Y', 'Z'
    else:
        raise ValueError("Invalid plane. Use 'xz' or 'yz'.")

    plt.figure(figsize=(8, 6))
    plt.imshow(slice_, cmap='viridis', aspect='auto', 
               extent=[0, slice_.shape[1], 0, slice_.shape[0]])
    plt.colorbar(label='Residual Intensity')
    plt.title(f'Residual Component in {plane.upper()} Plane')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def save_residual_slices(details, plane='xz', save_path='residual_plot.png',my_type='daa'):
    """
    Save residual components of the 3D wavelet transform as an image.

    Parameters:
        details (dict): Detail coefficients from the wavelet transform.
        plane (str): The plane to visualize ('xz' or 'yz').
        save_path (str): Path to save the plot.
    """
    residual = details[my_type]  # Example: Choose specific detail coefficient
    num_frames, height, width = residual.shape


    if plane == 'xz':
        # residual = details['ddd']  # Example: Choose specific detail coefficient
        # num_frames, height, width = residual.shape
        slice_ = residual[:, :, width // 2]  # Fixed y-axis
        xlabel, ylabel = 'X', 'Z'
    elif plane == 'yz':
        # residual = details['ddd']  # Example: Choose specific detail coefficient
        # num_frames, height, width = residual.shape
        slice_ = residual[:, height // 2, :]  # Fixed x-axis
        xlabel, ylabel = 'Y', 'Z'
    else:
        raise ValueError("Invalid plane. Use 'xz' or 'yz'.")

    plt.figure(figsize=(8, 6))
    plt.imshow(slice_, cmap='gray', aspect='auto', 
               extent=[0, slice_.shape[1], 0, slice_.shape[0]])
    plt.colorbar(label='Residual Intensity')
    plt.title(f'Residual Component in {plane.upper()} Plane')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path, dpi=300)
    plt.close()  # Close the plot to avoid displaying it


def plot_3d_wavelet_residual_fast(details, detail_key='daa', threshold=0.1, downsample=4, save_path='3d_residual_fast.png'):
    """
    Fast 3D plotting of wavelet residual components with downsampling.

    Parameters:
        details (dict): Wavelet detail coefficients.
        detail_key (str): The specific detail component to visualize (e.g., 'daa', 'dda').
        threshold (float): Value to threshold small residuals for cleaner plots.
        downsample (int): Factor by which to downsample the data.
        save_path (str): Path to save the 3D plot.
    """
    # Extract the residual component
    residual = details[detail_key]
    z_dim, y_dim, x_dim = residual.shape

    # Downsample the data for faster plotting
    residual_downsampled = residual[::downsample, ::downsample, ::downsample]
    z_downsampled, y_downsampled, x_downsampled = np.meshgrid(
        np.arange(0, z_dim, downsample),
        np.arange(0, y_dim, downsample),
        np.arange(0, x_dim, downsample),
        indexing='ij'
    )

    # Apply a threshold to filter small residual values for clarity
    mask = np.abs(residual_downsampled) > threshold
    x_vals = x_downsampled[mask]
    y_vals = y_downsampled[mask]
    z_vals = z_downsampled[mask]
    intensity = residual_downsampled[mask]

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        x_vals, y_vals, z_vals,
        c=intensity, cmap='viridis', s=5, alpha=0.8
    )
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title(f'3D Residual Plot for {detail_key.upper()} (Downsampled)')
    fig.colorbar(scatter, label='Residual Intensity', shrink=0.6)

    # Save the plot
    plt.savefig(save_path, dpi=300)
    plt.close()


def extract_wavelet_statistics(wavelet_coeffs):
    """
    Extract statistical features from wavelet coefficients (e.g., aaa, daa, ddd).
    Returns a dictionary with statistical features.
    """
    features = {}
    # wavelet_coeffs = wavelet_coeffs[f'{my_type}']
    
    # Compute mean, std, skewness, kurtosis, and entropy for the wavelet coefficients
    features['mean'] = np.mean(wavelet_coeffs)
    features['std'] = np.std(wavelet_coeffs)
    features['skewness'] = stats.skew(wavelet_coeffs.flatten())
    features['kurtosis'] = stats.kurtosis(wavelet_coeffs.flatten())
    features['entropy'] = stats.entropy(np.histogram(wavelet_coeffs.flatten(), bins=20)[0])
    features['energy'] = np.sum(wavelet_coeffs**2)

    return features


def save_features_to_csv(features_dict, filename):
    """
    Save the features to a CSV file.
    """
    # Convert features dictionary into a pandas DataFrame
    df = pd.DataFrame(features_dict)
    
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)


def temporal_difference_and_sum(video_array):
    temporal_difference = np.diff(video_array, axis=0)
    temporal_sum = video_array[:-1] + video_array[1:]
    return temporal_difference, temporal_sum


def temporal_approximation_difference(approximation):
    return np.diff(approximation, axis=0)


def ggd_pdf(x, alpha, beta):
    """
    Compute the GGD PDF for a given data array.
    """
    coefficient = beta / (2 * alpha * gamma(1 / beta))
    return coefficient * np.exp(-(np.abs(x) / alpha) ** beta)


def ggd_log_likelihood(params, data):
    """
    Compute the negative log-likelihood for GGD.
    """
    alpha, beta = params
    pdf = ggd_pdf(data, alpha, beta)
    log_likelihood = np.sum(np.log(pdf + 1e-8))  # Avoid log(0)
    return -log_likelihood


def fast_ggd_params(data):
    """
    Estimate the parameters of the Generalized Gaussian Distribution (GGD)
    using moments (mean, std, skewness, kurtosis).
    
    Parameters:
        data (numpy.ndarray): Input data array.
        
    Returns:
        tuple: Estimated parameters (alpha, beta).
    """
    # Calculate the standard deviation (scale parameter)
    alpha = np.std(data)
    
    # Calculate the skewness and kurtosis of the data
    data_skewness = stats.skew(data)
    data_kurtosis = stats.kurtosis(data)

    # Estimate the shape parameter (beta) using skewness and kurtosis
    # Here, we use an approximation based on kurtosis and skewness
    beta = data_kurtosis / (data_skewness ** 2) if data_skewness != 0 else 2

    return alpha, beta


def plot_ggd_from_params(alpha, beta, title="GGD Curve", save_path=None):
    """
    Plot a Generalized Gaussian Distribution curve given alpha and beta.

    Parameters:
        alpha (float): Scale parameter of GGD.
        beta (float): Shape parameter of GGD.
        title (str): Title of the plot.
        save_path (str): Path to save the plot (optional).
    """
    # Generate a range of x values
    x = np.linspace(-10 * alpha, 10 * alpha, 1000)

    # Compute GGD PDF
    ggd_pdf = gennorm.pdf(x, beta, scale=alpha)

    # Plot the GGD curve
    plt.figure(figsize=(8, 6))
    plt.plot(x, ggd_pdf, 'r-', label=f'GGD ($\\alpha={alpha:.2f}$, $\\beta={beta:.2f}$)')

    # Plot settings
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def ggd_mean_variance(alpha, beta):
    """
    Calculate the mean and variance of a Generalized Gaussian Distribution (GGD) 
    given the scale (alpha) and shape (beta) parameters.

    Parameters:
        alpha (float): Scale parameter of GGD.
        beta (float): Shape parameter of GGD.

    Returns:
        tuple: Mean and variance of the GGD.
    """
    # Mean of GGD
    mean = 0  # GGD is symmetric around 0

    # Variance of GGD
    variance = alpha**2 * (gamma(3 / beta) / gamma(1 / beta))
    
    return mean, variance


def plot_gaussian_curve(mean=0, std=1, title=None, save_path=None):
    # Generate x values
    x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)

    # Compute the PDF of the normal distribution
    gaussian_pdf = norm.pdf(x, loc=mean, scale=std)

    # Plot the Gaussian curve
    plt.figure(figsize=(8, 6))
    plt.plot(x, gaussian_pdf, 'b-', label=f'Gaussian ($\\mu={mean}$, $\\sigma={std}$)')

    # Plot settings
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend()

    # Save the plot if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def save_subband_video(subband, subband_name, plane,name,wavelet):
    # Normalize subband to 0-255 for visualization
    # subband = (subband - subband.min()) / (subband.max() - subband.min()) * 255
    subband = subband.astype(np.uint8)
    frame_rate = 30
    # Create a video writer for this subband
    output_path = os.path.join(f'subband_videos/{name}', f"{name}_{wavelet}_subband_{subband_name}_{plane}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    [frame_count,frame_height,frame_width] = subband.shape

    if plane == "xy":
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height), isColor=False)
        for i in range(frame_count):
            frame = subband[i, :, :]
            out.write(frame)

    elif plane == "xz":
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_count), isColor=False)
        for i in range(frame_height):
            frame = subband[:, i, :]
            out.write(frame)

    elif plane == "yz":
        out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_height, frame_count), isColor=False)
        for i in range(frame_width):
            frame = subband[:, :, i]
            out.write(frame)

    out.release()
    print(f"Saved subband video: {output_path}")


def plot_csv_results(file_path):
    data = pd.read_csv(file_path)

    # Set up visualization styles
    sns.set(style="whitegrid", palette="muted")

    # 1. Plot Comparisons
    # Line plot of mean values for each element
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=data, x="element", y="mean", hue="video_name", marker="o")
    plt.title("Mean Values by Element and Video")
    plt.xlabel("Element")
    plt.ylabel("Mean")
    plt.xticks(rotation=45)
    plt.legend(title="Video Name")
    plt.tight_layout()
    plt.show()

    # Bar plot of energy values for each video
    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x="video_name", y="energy", hue="element", ci=None)
    plt.title("Energy Values by Video and Element")
    plt.xlabel("Video Name")
    plt.ylabel("Energy")
    plt.xticks(rotation=45)
    plt.legend(title="Element")
    plt.tight_layout()
    plt.show()

    # 2. Statistical Analysis
    # Display summary statistics
    print("\nSummary Statistics:")
    print(data.describe())

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    correlation_matrix = data[['mean', 'std', 'skewness', 'kurtosis', 'entropy', 'energy']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # 3. Filtering and Grouping
    # Focus on a specific video or element
    selected_video = 'ant_db4'  # Change as needed
    selected_element = 'ant_temp_diff'  # Change as needed

    # Filter data by selected video
    filtered_data_video = data[data['video_name'] == selected_video]
    plt.figure(figsize=(12, 6))
    sns.barplot(data=filtered_data_video, x="element", y="energy", ci=None)
    plt.title(f"Energy Values for {selected_video}")
    plt.xlabel("Element")
    plt.ylabel("Energy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Filter data by selected element
    filtered_data_element = data[data['element'] == selected_element]
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=filtered_data_element, x="video_name", y="entropy", marker="o")
    plt.title(f"Entropy Values for {selected_element}")
    plt.xlabel("Video Name")
    plt.ylabel("Entropy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 4. Save Plots (Optional)
    # Uncomment the following to save figures to files:
    # plt.savefig(f"{element}_comparison_plot.png")


def analyze_results(name,subband,wavelet):
    video_path = fr'/home/atila/Desktop/comp_vis_proj/{name}.mp4'
    video_array = video_to_3d_array(video_path)
    approximation, details = apply_3d_wavelet_transform(video_array,wavelet=wavelet)
    residual_subband = details[subband]
    num_frames, height, width = residual_subband.shape

    for subbands in details:
        print(subbands)
        save_subband_video(details[subbands],subbands,"xy",name, wavelet)
        save_subband_video(details[subbands],subbands,"xz",name, wavelet)
        save_subband_video(details[subbands],subbands,"yz",name, wavelet)

    temp_diff, temp_sum = temporal_difference_and_sum(residual_subband) 
    # temporal_diff_xz= temp_diff[:,:,width//2]
    # temporal_diff_yz = temp_diff[:,height//2,:]
    # temporal_sum_xz= temp_sum[:,:,width//2]
    # temporal_sum_yz = temp_sum[:,height//2,:]

    approximation_diff = temporal_approximation_difference(approximation)
    # approximation_diff_xz= approximation_diff[:,:,width//2]
    # approximation_diff_yz = approximation_diff[:,height//2,:]

    temp_diff_flatten = temp_diff.flatten()
    temp_sum_flatten = temp_sum.flatten()
    approximation_diff_flatten = approximation_diff.flatten()

    diff_features = extract_wavelet_statistics(temp_diff_flatten)
    sum_features = extract_wavelet_statistics(temp_sum_flatten)
    app_features = extract_wavelet_statistics(approximation_diff_flatten)

    print(diff_features)
    print(sum_features)
    print(app_features)

    features_dict = {  # Dictionary to hold features for all videos
        'video_name': [], 
        'element': [],  
        'mean': [],
        'std': [],
        'skewness': [],
        'kurtosis': [],
        'entropy': [],
        'energy': []
    }

    features_dict['video_name'].append(f'{name}_{wavelet}')
    features_dict['element'].append(f'{name}_temp_diff')
    for key in diff_features:
            features_dict[key].append(diff_features[key])

    features_dict['video_name'].append(f'{name}_{wavelet}')
    features_dict['element'].append(f'{name}_temp_sum')
    for key in sum_features:
            features_dict[key].append(sum_features[key])

    features_dict['video_name'].append(f'{name}_{wavelet}')
    features_dict['element'].append(f'{name}_app')
    for key in app_features:
            features_dict[key].append(app_features[key])


    print('starting fitting GGD')
    alpha_diff, beta_diff = fast_ggd_params(temp_diff_flatten)
    alpha_sum, beta_sum = fast_ggd_params(temp_sum_flatten)
    alpha_app, beta_app = fast_ggd_params(approximation_diff_flatten)

    print(f'{name} alpha_diff = {alpha_diff}, beta_diff = {beta_diff}')
    print(f'{name} alpha_sum = {alpha_sum}, beta_sum = {beta_sum}')
    print(f'{name} alpha_app = {alpha_app}, beta_app = {beta_app}')

    mean_diff, var_diff = ggd_mean_variance(alpha_diff,beta_diff)
    mean_sum, var_sum = ggd_mean_variance(alpha_sum,beta_sum)
    mean_app, var_app = ggd_mean_variance(alpha_app,beta_app)

    plot_gaussian_curve(mean_diff,var_diff,'temp_diff',f'gaussian/{name}_temp_diff.png')
    plot_gaussian_curve(mean_sum,var_sum,'temp_sum',f'gaussian/{name}_temp_sum.png')
    plot_gaussian_curve(mean_app,var_app,'app_diff',f'gaussian/{name}_app_diff.png')

    plot_ggd_from_params(alpha_diff,beta_diff,'Temp Difference',f'ggd_plots/{name}_temporal_diff_ggd.png')
    plot_ggd_from_params(alpha_sum,beta_sum,'Temp Sum',f'ggd_plots/{name}_temporal_sum_ggd.png')
    plot_ggd_from_params(alpha_app,beta_app,'Temp Difference',f'ggd_plots/{name}_app_ggd.png')
    

    # plt.figure(figsize=(8, 6))
    # plt.imshow(temporal_diff_xz, cmap='gray', aspect='auto', 
    #             extent=[0, temporal_diff_xz.shape[1], 0, temporal_diff_xz.shape[0]])
    # plt.colorbar(label='Residual Intensity')
    # plt.title(f'Residual Difference in xz Plane')
    # plt.xlabel('X')
    # plt.ylabel('Z')
    # plt.savefig(f'difference/{name}_xz_temporal_diff_ddd_{wavelet}.png', dpi=300)
    # plt.close()  # Close the plot to avoid displaying it

    # plt.figure(figsize=(8, 6))
    # plt.imshow(temporal_diff_yz, cmap='gray', aspect='auto', 
    #             extent=[0, temporal_diff_yz.shape[1], 0, temporal_diff_yz.shape[0]])
    # plt.colorbar(label='Residual Intensity')
    # plt.title(f'Residual Difference in yz Plane')
    # plt.xlabel('Y')
    # plt.ylabel('Z')
    # plt.savefig(f'difference/{name}_yz_temporal_diff_ddd_{wavelet}.png', dpi=300)
    # plt.close()  # Close the plot to avoid displaying it

    # plt.figure(figsize=(8, 6))
    # plt.imshow(temporal_sum_xz, cmap='gray', aspect='auto', 
    #             extent=[0, temporal_sum_xz.shape[1], 0, temporal_sum_xz.shape[0]])
    # plt.colorbar(label='Residual Intensity')
    # plt.title(f'Residual Sum in xz Plane')
    # plt.xlabel('X')
    # plt.ylabel('Z')
    # plt.savefig(f'sum/{name}_xz_temporal_sum_ddd_{wavelet}.png', dpi=300)
    # plt.close()  # Close the plot to avoid displaying it

    # plt.figure(figsize=(8, 6))
    # plt.imshow(temporal_sum_yz, cmap='gray', aspect='auto', 
    #             extent=[0, temporal_sum_yz.shape[1], 0, temporal_sum_yz.shape[0]])
    # plt.colorbar(label='Residual Intensity')
    # plt.title(f'Residual Sum in yz Plane')
    # plt.xlabel('Y')
    # plt.ylabel('Z')
    # plt.savefig(f'sum/{name}_yz_temporal_sum_ddd_{wavelet}.png', dpi=300)
    # plt.close()  # Close the plot to avoid displaying it

    # plt.figure(figsize=(8, 6))
    # plt.imshow(approximation_diff_xz, cmap='gray', aspect='auto', 
    #             extent=[0, approximation_diff_xz.shape[1], 0, approximation_diff_xz.shape[0]])
    # plt.colorbar(label='Residual Intensity')
    # plt.title(f'Approximation Difference in xz Plane')
    # plt.xlabel('X')
    # plt.ylabel('Z')
    # plt.savefig(f'approximation/{name}_xz_approximation_diff_{wavelet}.png', dpi=300)
    # plt.close()  # Close the plot to avoid displaying it

    # plt.figure(figsize=(8, 6))
    # plt.imshow(approximation_diff_yz, cmap='gray', aspect='auto', 
    #             extent=[0, approximation_diff_yz.shape[1], 0, approximation_diff_yz.shape[0]])
    # plt.colorbar(label='Residual Intensity')
    # plt.title(f'Approximation Difference in yz Plane')
    # plt.xlabel('Y')
    # plt.ylabel('Z')
    # plt.savefig(f'approximation/{name}_yz_approximation_diff_{wavelet}.png', dpi=300)
    # plt.close()  # Close the plot to avoid displaying it

    return features_dict

def main():
    name_list = ['ant','duck','misato','referee']
    wavelet = 'sym2'

    features_dict = {  # Dictionary to hold features for all videos
        'video_name': [],   
        'element': [],
        'mean': [],
        'std': [],
        'skewness': [],
        'kurtosis': [],
        'entropy': [],
        'energy': []
    }

    for names in name_list:
        append_dict = analyze_results(names,'ddd',wavelet)
        for key in features_dict.keys():
            features_dict[key].extend(append_dict[key]) 
        print(f'extracted {names}')
    save_features_to_csv(features_dict,f'video_features_{wavelet}.csv')

    plot_csv_results(f"video_features_{wavelet}.csv")



if __name__ == '__main__':
    main()