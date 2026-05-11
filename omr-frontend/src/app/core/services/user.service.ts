import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})

export class UserService {

  private apiUrl = 'http://localhost:5000/admin/users';

  constructor(private http: HttpClient) {}

  getUsers() {
    return this.http.get(
      this.apiUrl,
      {
        withCredentials: true
      }
    );
  }

  createUser(data: any) {

    return this.http.post(
      this.apiUrl,
      data,
      {
        withCredentials: true
      }
    );
  }

  updateUser(userId: number, data: any) {
    return this.http.put(
      `${this.apiUrl}/${userId}`,
      data, {withCredentials: true}
    );
  }

  deleteUser(userId: number) {
    return this.http.delete(
      `${this.apiUrl}/${userId}`,
      {
        withCredentials: true
      }
    );
  }
}