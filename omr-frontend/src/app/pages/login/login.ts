import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { AuthService } from '../../core/services/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './login.html',
  styleUrls: ['./login.scss']
})
export class Login {
  email = '';
  password = '';
  loading = false;
  errorMessage = '';
  showPassword = false;

  constructor(
    private router: Router,
    private authService: AuthService,
    private cdr: ChangeDetectorRef
  ) {}

  login() {
    this.loading = true;
    this.errorMessage = '';

    const payload = { email: this.email, password: this.password };

    this.authService.login(payload).subscribe({
      next: (response: any) => {
        this.loading = false;
        this.router.navigate(['/analyzer']);
      },
      error: (error) => {
        this.loading = false;
        this.errorMessage =
          error?.error?.msg ||
          error?.error?.message ||
          'Invalid email or password. Please try again.';
        this.cdr.detectChanges();
      }
    });
  }
}